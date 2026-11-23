"""Load experts-only ModelOpt NVFP4 and preserve every other source dtype."""
from contextlib import ExitStack
import json
import logging
import os
from pathlib import Path
import re

import torch
from safetensors import safe_open

from ..load import _load_into, checkpoint_files
from ...kernels.nvfp4 import NvFp4Experts

log = logging.getLogger('flide.nvfp4')
DTYPES = {'F32': torch.float32, 'BF16': torch.bfloat16, 'F16': torch.float16}


def validate(model_dir, device, dtype, loader, tp, options):
    path = Path(model_dir) / 'hf_quant_config.json'
    q = json.loads(path.read_text()).get('quantization', {}) if path.is_file() else {}
    if q.get('quant_algo') != 'NVFP4' or q.get('group_size') != 16:
        raise ValueError('NVFP4 requires experts-only ModelOpt metadata with group_size=16')
    if loader != 'direct' or tp is not None or dtype != 'bfloat16' or not str(device).startswith('cuda'):
        raise ValueError('NVFP4 requires the direct loader, a single CUDA GPU and BF16 activations')
    if options.get('float32_vision'):
        raise ValueError('NVFP4 preserves source vision dtype; float32_vision is incompatible')
    if (Path(model_dir) / 'lora').exists():
        raise ValueError('NVFP4 with LoRA is not supported')
    if torch.cuda.get_device_capability(device) != (12, 0):
        raise ValueError('The Triton NVFP4 backend is currently validated on SM120 only')


def mapped(key, rules):
    for pattern, replacement in rules:
        key = pattern.sub(replacement, key)
    return key


def checked(tensor, shape, dtype, key):
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
        raise ValueError(f'{key}: expected {tuple(shape)} {dtype}, got {tuple(tensor.shape)} {tensor.dtype}')
    return tensor


@torch.inference_mode()
def load(model, model_dir, device, key_map, skip):
    rules = [(re.compile(a), b) for a, b in key_map]
    parameters = dict(model.named_parameters())
    expert_shapes = {name: tuple(p.shape) for name, p in parameters.items()
                     if name.endswith('.experts.gate_up_proj')}
    if not expert_shapes:
        raise ValueError('NVFP4 requires Qwen3.5 routed experts')
    prefixes = tuple(name.removesuffix('.gate_up_proj') + '.' for name in expert_shapes)
    store, preserved = {}, {}
    with ExitStack() as stack:
        files = [stack.enter_context(safe_open(str(p), framework='pt', device='cpu'))
                 for p in checkpoint_files(model_dir)]
        tensors = {}
        for f in files:
            for key in f.keys():
                name = mapped(key, rules)
                if name in tensors:
                    raise ValueError(f'duplicate checkpoint tensor: {name}')
                tensors[name] = (f, key)
        for name, (f, key) in tensors.items():
            if name.startswith(skip) or name.startswith(prefixes):
                continue
            p = parameters.get(name)
            if p is None:
                raise ValueError(f'Unexpected tensor outside routed experts: {key}')
            dtype = DTYPES.get(f.get_slice(key).get_dtype())
            if dtype is None:
                raise ValueError(f'Non-expert tensor must retain source floating dtype: {key}')
            if p.dtype != dtype:
                p.data = torch.empty(p.shape, device=device, dtype=dtype)
            preserved[name] = (f, key, dtype)
        _load_into(model, model_dir, device, key_map, skip + prefixes, None, None, None)
        rows = []
        for name, (f, key, dtype) in preserved.items():
            p = parameters[name]
            if p.dtype != dtype:
                raise ValueError(f'Source dtype changed during loading: {key}')
            if dtype == torch.float32:
                equal = torch.equal(p.detach().cpu().contiguous().view(torch.uint8),
                                    f.get_tensor(key).contiguous().view(torch.uint8))
                if not equal:
                    raise ValueError(f'Source FP32 bytes changed: {key}')
                rows.append({'source_name': key, 'runtime_name': name, 'dtype': str(dtype), 'bits_equal': equal})
        for fused, (e, ni, h) in expert_shapes.items():
            prefix, i = fused.removesuffix('.gate_up_proj'), ni // 2
            if ni % 2 or i % 16 or h % 16:
                raise ValueError(f'Unsupported NVFP4 expert shape: {fused}')
            expected = set()

            def read(projection, field, shape, dtype):
                parts = []
                for expert in range(e):
                    name = f'{prefix}.{expert}.{projection}.{field}'
                    expected.add(name)
                    if name not in tensors:
                        raise ValueError(f'Missing NVFP4 tensor: {name}')
                    f, key = tensors[name]
                    tensor = checked(f.get_tensor(key), shape, dtype, key)
                    if field in ('weight_scale_2', 'input_scale'):
                        if not torch.isfinite(tensor).all() or not (tensor > 0).all():
                            raise ValueError(f'Invalid positive FP32 scale: {key}')
                    parts.append(tensor)
                return torch.stack(parts).to(device)

            gate, up = [read(p, 'weight', (i, h // 2), torch.uint8) for p in ('gate_proj', 'up_proj')]
            down = read('down_proj', 'weight', (h, i // 2), torch.uint8)
            gs, us = [read(p, 'weight_scale', (i, h // 16), torch.float8_e4m3fn) for p in ('gate_proj', 'up_proj')]
            ds = read('down_proj', 'weight_scale', (h, i // 16), torch.float8_e4m3fn)
            gg, ug, dg = [read(p, 'weight_scale_2', (), torch.float32) for p in ('gate_proj', 'up_proj', 'down_proj')]
            ga, ua, da = [read(p, 'input_scale', (), torch.float32) for p in ('gate_proj', 'up_proj', 'down_proj')]
            actual = {name for name in tensors if name.startswith(prefix + '.')}
            if actual != expected:
                raise ValueError(f'Unsupported NVFP4 tensors: {sorted(actual - expected)[:5]}')
            store[prefix] = NvFp4Experts(torch.cat([gate, up], dim=1).contiguous(), down,
                                        torch.cat([gs, us], dim=1).contiguous(), ds,
                                        torch.stack([gg, ug], dim=1), dg, torch.maximum(ga.max(), ua.max()), da.max())
            log.info('NVFP4 loaded %s (%d experts)', prefix, e)
    audit = {'complete': True, 'checkpoint': str(Path(model_dir).resolve()),
             'non_expert_tensors_checked': len(preserved), 'fp32_tensors_checked': len(rows),
             'all_source_fp32_dtype_and_bits_match': True, 'parameters': rows}
    if path := os.environ.get('FLIDE_DTYPE_AUDIT'):
        Path(path).write_text(json.dumps(audit, indent=2) + '\n')
    log.info('SOURCE_DTYPE_RUNTIME_AUDIT_PASS: %d FP32 tensors; %d source dtype checks', len(rows), len(preserved))
    return store, audit
