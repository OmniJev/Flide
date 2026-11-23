"""Checkpoint layout compatibility, independent of any particular model release."""
import json
import sys
from pathlib import Path
import tempfile

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.backbones.load import FP8CheckpointError, load_into


def write(path, tensors, separate_scales=False):
    parts = [tensors]
    if separate_scales:
        parts = [{k: v for k, v in tensors.items() if not k.endswith('_scale')},
                 {k: v for k, v in tensors.items() if k.endswith('_scale')}]
    index = {}
    for i, part in enumerate(parts):
        name = f'part-{i}.safetensors'
        save_file(part, str(path / name))
        index.update({k: name for k in part})
    (path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': index}))


def dense(meta=False):
    m = torch.nn.Module()
    m.proj = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16, device='meta' if meta else 'cpu')
    return m


def experts(meta=False):
    m = torch.nn.Module()
    m.mlp = torch.nn.Module()
    m.mlp.experts = torch.nn.Module()
    device = 'meta' if meta else 'cpu'
    for name, shape in [('gate_up_proj', (2, 6, 4)), ('down_proj', (2, 4, 3))]:
        m.mlp.experts.register_parameter(name, torch.nn.Parameter(torch.empty(shape, dtype=torch.bfloat16, device=device)))
    return m


def exact(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_unquantized(path):
    m = dense()
    weight = torch.arange(12).reshape(3, 4).to(torch.bfloat16)
    write(path, {'old.proj.weight': weight, 'mtp.weight': torch.ones(1)})
    load_into(m, str(path), 'cpu', key_map=[(r'^old\.', '')], skip=('mtp.',))
    exact(m.proj.weight, weight)
    m = experts()
    fused = {k: torch.arange(p.numel()).reshape(p.shape).to(torch.bfloat16) for k, p in m.named_parameters()}
    write(path, fused)
    load_into(m, str(path), 'cpu')
    for k, p in m.named_parameters():
        exact(p, fused[k])
    split = {}
    for e in range(2):
        split[f'mlp.experts.{e}.gate_proj.weight'] = fused['mlp.experts.gate_up_proj'][e, :3].clone()
        split[f'mlp.experts.{e}.up_proj.weight'] = fused['mlp.experts.gate_up_proj'][e, 3:].clone()
        split[f'mlp.experts.{e}.down_proj.weight'] = fused['mlp.experts.down_proj'][e].clone()
    write(path, split)
    m = experts()
    load_into(m, str(path), 'cpu')
    for k, p in m.named_parameters():
        exact(p, fused[k])
    print('PASS BF16 dense/key mapping/skips, fused MoE, split MoE')


def test_dense_fp8(path):
    q = torch.arange(12).reshape(3, 4).to(torch.float8_e4m3fn)
    for scale in [torch.tensor(.125), torch.tensor([.125, .25, .5]), torch.tensor([[.125], [.25], [.5]])]:
        write(path, {'proj.weight': q, 'proj.weight_scale': scale}, separate_scales=True)
        store = {}
        load_into(dense(True), str(path), 'cpu', lean=lambda n: n == 'proj.weight', fp8=store)
        actual, scales = store['proj.weight']
        exact(actual, q)
        want = torch.full((3, 1), .125) if scale.numel() == 1 else scale.reshape(3, 1)
        assert torch.equal(scales, want)
    # A scalar tensor whose name ends in _scale must still load as a normal buffer.
    m = dense()
    m.register_buffer('temperature', torch.tensor(0.))
    m.register_buffer('temperature_scale', torch.tensor(0.))
    write(path, {'proj.weight': q.to(torch.bfloat16), 'temperature': torch.tensor(2.), 'temperature_scale': torch.tensor(.75)})
    load_into(m, str(path), 'cpu')
    assert m.temperature_scale.item() == .75 and m.temperature.item() == 2.
    print('PASS dense FP8 scalar/row scales across shards; ordinary scale buffers preserved')


def test_fused_fp8(path):
    m = experts(True)
    qs = {k: torch.arange(p.numel()).reshape(p.shape).to(torch.float8_e4m3fn) for k, p in m.named_parameters()}
    for layout in ['scalar', 'expert', 'row', 'gate_up_shards']:
        tensors, expected = dict(qs), {}
        for name, q in qs.items():
            e, n, _ = q.shape
            if layout == 'scalar':
                scale, want = torch.tensor(.125), torch.full((e, n), .125)
            elif layout == 'expert':
                scale = torch.tensor([.125, .25])
                want = torch.stack([torch.full((n,), .125), torch.full((n,), .25)])
            elif layout == 'gate_up_shards' and name.endswith('gate_up_proj'):
                scale = torch.tensor([[.125, .25], [.5, 1.]])
                want = torch.tensor([[.125] * 3 + [.25] * 3, [.5] * 3 + [1.] * 3])
            else:
                scale = torch.arange(1, e * n + 1).float().reshape(e, n) / 8
                want = scale.clone()
            tensors[name + '_scale'] = scale
            expected[name] = want
        write(path, tensors, separate_scales=True)
        store = {}
        result = load_into(m, str(path), 'cpu', lean=lambda n: True, fp8=store)
        assert result['unused'] == []
        for name, q in qs.items():
            actual, scale = store[name]
            exact(actual, q)
            assert torch.equal(scale, expected[name])
    print('PASS fused FP8 scalar/expert/row/gate-up scales; exact bytes and cross-shard scales')


def test_unsupported_scales(path):
    q = torch.arange(12).reshape(3, 4).to(torch.float8_e4m3fn)
    for tensors in [{'proj.weight': q, 'proj.weight_scale_inv': torch.ones(2, 2)},
                    {'proj.weight': q, 'proj.weight_scale': torch.ones(2, 2)},
                    {'proj.weight': q.float().to(torch.float8_e5m2), 'proj.weight_scale': torch.tensor(1.)}]:
        write(path, tensors)
        try:
            load_into(dense(True), str(path), 'cpu', lean=lambda n: True, fp8={})
        except FP8CheckpointError:
            pass
        else:
            raise AssertionError('unsupported FP8 layout was silently accepted')
    print('PASS unsupported block/inverse scales and FP8 dtypes fail explicitly')


def test_unindexed_shards_and_duplicates(path):
    q = torch.arange(12).reshape(3, 4).to(torch.float8_e4m3fn)
    write(path, {'proj.weight': q, 'proj.weight_scale': torch.tensor(.125)}, separate_scales=True)
    (path / 'model.safetensors.index.json').unlink()
    store = {}
    load_into(dense(True), str(path), 'cpu', lean=lambda name: True, fp8=store)
    exact(store['proj.weight'][0], q)
    assert torch.equal(store['proj.weight'][1], torch.full((3, 1), .125))
    save_file({'proj.weight': q}, str(path / 'part-2.safetensors'))
    try:
        load_into(dense(True), str(path), 'cpu', lean=lambda name: True, fp8={})
    except ValueError as exc:
        assert 'duplicate checkpoint tensor' in str(exc)
    else:
        raise AssertionError('duplicate weights across unindexed shards were accepted')
    print('PASS unindexed cross-shard scales and duplicate tensor rejection')


def test_dequantized_and_quantized_weights(path):
    m = experts()
    weights, expected = {}, {}
    for name, parameter in m.named_parameters():
        q = torch.arange(parameter.numel()).reshape(parameter.shape).to(torch.float8_e4m3fn)
        scale = torch.arange(1, q.shape[0] * q.shape[1] + 1).reshape(q.shape[:2]).float() / 8
        weights[name], weights[name + '_scale'] = q, scale
        expected[name] = (q.float() * scale.unsqueeze(-1)).to(torch.bfloat16)
    write(path, weights, separate_scales=True)
    load_into(m, str(path), 'cpu', lean=lambda name: False)
    for name, parameter in m.named_parameters():
        exact(parameter, expected[name])
    from flide.kernels.moe import quantize_experts

    write(path, expected)
    store = {}
    load_into(experts(True), str(path), 'cpu', lean=lambda name: True, fp8=store)
    for name, value in expected.items():
        q, scale = quantize_experts(value)
        exact(store[name][0], q)
        assert torch.equal(store[name][1], scale)
    print('PASS 3D FP8 dequantization and BF16 expert quantization')


def test_tensor_parallel_fp8_scales(path):
    q = torch.arange(12).reshape(3, 4).to(torch.float8_e4m3fn)
    for scale in (torch.tensor(.125), torch.tensor([[.125], [.25], [.5]])):
        write(path, {'proj.weight': q, 'proj.weight_scale': scale}, separate_scales=True)
        full_scale = scale.reshape(-1, 1).expand(3, 1)
        for dim in (0, 1):
            for to_fp8 in (False, True):
                store, held = {}, {}
                index = slice(1, 3)
                cut = lambda name: (dim, index, to_fp8)
                load_into(dense(True), str(path), 'cpu', lean=lambda name: True, fp8=store, cut=cut, held=held)
                if to_fp8:
                    expected_q = q[index] if dim == 0 else q[:, index]
                    expected_scale = full_scale[index] if dim == 0 else full_scale
                    exact(store['proj.weight'][0], expected_q)
                    assert torch.equal(store['proj.weight'][1], expected_scale)
                    assert not held
                else:
                    value = (q.float() * full_scale).to(torch.bfloat16)
                    exact(held['proj.weight'], value[index] if dim == 0 else value[:, index])
                    assert not store
    print('PASS existing FP8 tensor-parallel row/column cuts and held weights')


def test_invalid_split_layouts(path):
    weights = {}
    for e in range(2):
        for projection in ('gate', 'up', 'down'):
            shape = (4, 3) if projection == 'down' else (3, 4)
            weights[f'mlp.experts.{e}.{projection}_proj.weight'] = torch.ones(shape, dtype=torch.bfloat16)
    cases = []
    bad_index = dict(weights)
    bad_index['mlp.experts.2.gate_proj.weight'] = bad_index.pop('mlp.experts.1.gate_proj.weight')
    cases.append((bad_index, 'invalid expert index'))
    bad_shape = dict(weights)
    bad_shape['mlp.experts.0.gate_proj.weight'] = torch.ones(4, 4, dtype=torch.bfloat16)
    cases.append((bad_shape, 'checkpoint shape'))
    duplicate = dict(weights)
    duplicate['mlp.experts.00.gate_proj.weight'] = duplicate['mlp.experts.0.gate_proj.weight'].clone()
    cases.append((duplicate, 'duplicate expert projection'))
    mixed = dict(weights)
    mixed['mlp.experts.gate_up_proj'] = torch.ones(2, 6, 4, dtype=torch.bfloat16)
    cases.append((mixed, 'both fused and split'))
    for tensors, message in cases:
        write(path, tensors)
        try:
            load_into(experts(), str(path), 'cpu')
        except ValueError as exc:
            assert message in str(exc), str(exc)
        else:
            raise AssertionError(f'invalid layout accepted: {message}')
    print('PASS invalid expert indices, shapes, duplicate projections and mixed layouts')


def main():
    for test in [test_unquantized, test_dense_fp8, test_fused_fp8, test_unsupported_scales,
                 test_unindexed_shards_and_duplicates, test_dequantized_and_quantized_weights,
                 test_tensor_parallel_fp8_scales, test_invalid_split_layouts]:
        with tempfile.TemporaryDirectory() as directory:
            test(Path(directory))


if __name__ == '__main__':
    main()
