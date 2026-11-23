"""Mixed source dtypes and packed experts survive the native NVFP4 loader."""
import json
from pathlib import Path
import tempfile

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

from flide.backbones.qwen35.decoder import Qwen35
from flide.formats.base import FINAL, Read
from flide.runtime.pack import Job, pack


@torch.inference_mode()
def main():
    cfg = Qwen3_5MoeTextConfig(vocab_size=512, hidden_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, moe_intermediate_size=64,
        shared_expert_intermediate_size=64, num_experts=4, num_experts_per_tok=2,
        layer_types=['full_attention'], pad_token_id=0, bos_token_id=1, eos_token_id=2,
        max_position_embeddings=512, rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
        'partial_rotary_factor': .25, 'mrope_section': [2, 1, 1], 'mrope_interleaved': True})
    model = Qwen3_5MoeForCausalLM(cfg).bfloat16()
    norm = model.model.layers[0].input_layernorm.weight
    norm.data = torch.linspace(.001001, .099999, norm.numel(), dtype=torch.float32)
    norm.data[0] = -0.0
    weights = {k: p.clone().contiguous() for k, p in model.state_dict().items()}
    for name in list(weights):
        if not name.endswith(('.experts.gate_up_proj', '.experts.down_proj')):
            continue
        dense = weights.pop(name)
        prefix, param = name.rsplit('.', 1)
        projections = [('down_proj', dense.shape[1])] if param == 'down_proj' else [('gate_proj', dense.shape[1] // 2), ('up_proj', dense.shape[1] // 2)]
        for expert in range(dense.shape[0]):
            for projection, n in projections:
                key = f'{prefix}.{expert}.{projection}.'
                k = dense.shape[2]
                weights[key + 'weight'] = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
                weights[key + 'weight_scale'] = torch.ones(n, k // 16, dtype=torch.float8_e4m3fn)
                weights[key + 'weight_scale_2'] = torch.tensor(.01)
                weights[key + 'input_scale'] = torch.tensor(.02)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        config = cfg.to_dict()
        config['architectures'] = ['Qwen3_5MoeForCausalLM']
        config['quantization_config'] = {'quant_method': 'modelopt', 'quant_algo': 'NVFP4'}
        (path / 'config.json').write_text(json.dumps(config))
        (path / 'hf_quant_config.json').write_text(json.dumps({'quantization': {'quant_algo': 'NVFP4', 'group_size': 16}}))
        save_file(weights, str(path / 'model.safetensors'))
        bb = Qwen35.load(directory, quantize='nvfp4', cuda_graphs=True)
        runtime_norm = bb.layers[0].input_layernorm.weight
        assert runtime_norm.dtype == torch.float32
        assert torch.equal(runtime_norm.cpu().view(torch.uint8), norm.view(torch.uint8))
        assert bb.layers[0].mlp.experts.gate_up_proj.is_meta
        assert bb.dtype_audit['fp32_tensors_checked'] == 1
        assert bb._w[0]['nvfp4'].w13.dtype == torch.uint8
        batch = pack([Job([1, 5, 9], [[4, 5]], [(Read(None),)])], torch.device('cuda'))
        hidden = bb.forward(batch)[FINAL]
        assert torch.isfinite(hidden).all()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = bb.forward(batch)[FINAL]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, hidden)
        # Replay must consume staged inputs rather than retain the first result.
        batch.input_ids.copy_(torch.tensor([1, 6, 10, 4, 7], device='cuda'))
        expected = bb.forward(batch)[FINAL].clone()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, expected)
        del weights['model.layers.0.mlp.experts.0.down_proj.input_scale']
        save_file(weights, str(path / 'model.safetensors'))
        try:
            Qwen35.load(directory, quantize='nvfp4', cuda_graphs=False)
        except ValueError as exc:
            assert 'Missing NVFP4 tensor' in str(exc)
        else:
            raise AssertionError('Incomplete calibration scales accepted')
    print('PASS source FP32 bytes including signed zero, packed expert allocation, native forward, graph replay with changed inputs and missing scales')


if __name__ == '__main__':
    main()
