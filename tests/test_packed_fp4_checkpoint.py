"""Reject packed NVFP4 before allocating a BF16 model or reading its weight bytes."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from flide.backbones.load import PackedFP4CheckpointError, check_checkpoint_quantization, load_into
from flide.backbones.qwen35.decoder import Qwen35


def expect_rejected(call):
    try:
        call()
    except PackedFP4CheckpointError as error:
        assert 'NVFP4' in str(error) and 'MoE backend' in str(error)
    else:
        raise AssertionError('packed NVFP4 reached the BF16/FP8 load path')


def main():
    layouts = [
        ('config.json', {'quantization_config': {'quant_method': 'modelopt', 'quant_algo': 'NVFP4'}}),
        ('hf_quant_config.json', {'quantization': {'quant_algo': 'NVFP4', 'group_size': 16}}),
        ('config.json', {'quantization_config': {'quant_algo': 'MIXED_PRECISION', 'quantized_layers': {
            'model.layers.0.mlp.experts': {'quant_algo': 'NVFP4', 'group_size': 16},
            'model.layers.0.self_attn.q_proj': {'quant_algo': 'FP8'}}}}),
    ]
    for filename, payload in layouts:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / filename).write_text(json.dumps(payload))
            model = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
            before = model.weight.detach().clone()
            with patch('safetensors.safe_open', side_effect=AssertionError('weight bytes were read')):
                expect_rejected(lambda: load_into(model, directory, 'cpu'))
            assert torch.equal(model.weight, before)
            with patch('transformers.AutoConfig.from_pretrained', side_effect=AssertionError('HF load reached')):
                for loader in ['direct', 'hf']:
                    for quantize in [None, 'fp8']:
                        expect_rejected(lambda: Qwen35.load(directory, device='cuda:0', loader=loader,
                                                            quantize=quantize))
    print('PASS NVFP4 and mixed-NVFP4 fail before model allocation, weight reads or HF fallback')
    for payload in [{}, {'quantization_config': {'quant_method': 'fp8'}},
                    {'quantization': {'quant_algo': 'FP8'}},
                    {'quantization_config': {'quant_algo': 'FP8', 'quantized_layers': None}},
                    {'quantization': {'quant_algo': 'MIXED_PRECISION', 'quantized_layers': {
                        'model.layers.0.self_attn.q_proj': {'quant_algo': 'FP8'}}}}]:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'config.json').write_text(json.dumps(payload))
            check_checkpoint_quantization(directory)
    print('PASS BF16 and FP8 metadata are unchanged')


if __name__ == '__main__':
    main()
