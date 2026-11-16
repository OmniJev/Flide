"""Split MoE checkpoints: expert ordering, FP8 scales, and incomplete weights."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile

import torch
from safetensors.torch import save_file

from flide.backbones.load import blank, load_into


def model():
    m = torch.nn.Module()
    m.mlp = torch.nn.Module()
    m.mlp.experts = torch.nn.Module()
    m.mlp.experts.gate_up_proj = torch.nn.Parameter(torch.full((2, 6, 4), float('nan'), dtype=torch.bfloat16))
    m.mlp.experts.down_proj = torch.nn.Parameter(torch.full((2, 4, 3), float('nan'), dtype=torch.bfloat16))
    return m


def checkpoint(path, missing=None, missing_scale=False):
    shards = [{}, {}]
    expected = {}
    for e in range(2):
        for j, projection in enumerate(('gate', 'up', 'down')):
            key = f'mlp.experts.{e}.{projection}_proj.weight'
            if key == missing:
                continue
            rows, cols = (4, 3) if projection == 'down' else (3, 4)
            value = (torch.arange(rows * cols).reshape(rows, cols).float() + e * 16 + j * 4).to(torch.float8_e4m3fn)
            scale = torch.tensor(0.125) if j == 0 else torch.arange(1, rows + 1).float().reshape(rows, 1) / 8
            expected[e, projection] = (value, scale.reshape(-1).expand(rows))
            shard = shards[j % 2]
            shard[key] = value
            if not missing_scale:
                shard[key + '_scale'] = scale
    index = {}
    for i, shard in enumerate(shards):
        name = f'part-{i}.safetensors'
        save_file(shard, str(path / name))
        index.update({k: name for k in shard})
    (path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': index}))
    return expected


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        expected = checkpoint(path)
        m = model().to('meta')
        store = {}
        lean = lambda name: name.endswith(('.experts.gate_up_proj', '.experts.down_proj'))
        result = load_into(m, str(path), 'cpu', lean=lean, fp8=store)
        assert result['unused'] == []
        for e in range(2):
            for projection, parameter, rows in (('gate', 'gate_up_proj', slice(0, 3)),
                                                 ('up', 'gate_up_proj', slice(3, None)),
                                                 ('down', 'down_proj', slice(None))):
                q, scale = store['mlp.experts.' + parameter]
                want_q, want_scale = expected[e, projection]
                assert torch.equal(q[e, rows].view(torch.uint8), want_q.view(torch.uint8))
                assert torch.equal(scale[e, rows], want_scale)
        assert all(p.is_meta for p in m.parameters())
        print('PASS exact FP8 bytes, expert order, gate/up concatenation, scalar/row scales, meta allocation')
        from flide.backbones.qwen35.decoder import Qwen35
        fake = SimpleNamespace(layers=[SimpleNamespace(mlp=m.mlp)], _w=[{'experts': None}])
        Qwen35._fp8_from_store(fake, store, {id(m.mlp.experts): 'mlp.experts'})
        assert not store and fake.fp8
        assert fake._w[0]['experts'].dtype == torch.float8_e4m3fn
        assert fake._w[0]['experts_scale'].shape == (2, 6)
        assert fake._w[0]['experts_down_scale'].shape == (2, 4)
        print('PASS native MoE weights and scales are attached without re-quantization')
        try:
            load_into(model(), str(path), 'cpu')
        except ValueError as exc:
            assert 'stored in float8' in str(exc)
        else:
            raise AssertionError('FP8 was accepted without explicit dequantization')
        print('PASS FP8 requires explicit FP8 loading')
        checkpoint(path, missing='mlp.experts.1.up_proj.weight')
        try:
            load_into(model().to('meta'), str(path), 'cpu', lean=lean, fp8={})
        except ValueError as exc:
            assert 'missing expert projections' in str(exc)
        else:
            raise AssertionError('incomplete expert was accepted')
        print('PASS incomplete fused expert is rejected')
        checkpoint(path, missing_scale=True)
        try:
            load_into(model().to('meta'), str(path), 'cpu', lean=lean, fp8={})
        except Exception as exc:
            assert 'scale' in str(exc)
        else:
            raise AssertionError('missing FP8 scale was accepted')
        print('PASS missing FP8 scale is rejected')
        empty_store = {}
        blank(model().to('meta'), 'cpu', lean=lean, fp8=empty_store)
        assert empty_store['mlp.experts.gate_up_proj'][1].shape == (2, 6)
        assert empty_store['mlp.experts.down_proj'][1].shape == (2, 4)
        print('PASS no-weight kernel warmup uses the native MoE scale layout')


if __name__ == '__main__':
    main()
