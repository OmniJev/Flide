"""The same tiny Qwen MoE model loaded from BF16 and equivalent FP8 layouts."""
import copy
import json
import sys
from pathlib import Path
import tempfile

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import Qwen3_5MoeForCausalLM
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from flide.backbones.qwen35.decoder import Qwen35
from flide.formats.base import FINAL, Read
from flide.kernels.moe import quantize_experts
from flide.runtime.pack import Job, pack


def write(path, config, weights, fp8=False):
    path.mkdir()
    data = config.to_dict()
    data['architectures'] = ['Qwen3_5MoeForCausalLM']
    if fp8:
        data['quantization_config'] = {'quant_method': 'fp8', 'activation_scheme': 'dynamic'}
    (path / 'config.json').write_text(json.dumps(data))
    save_file(weights, str(path / 'model.safetensors'))


def forward(backbone):
    job = Job([1, 5, 9, 7, 2], [[4, 5], [6, 7, 8]], [(Read(None),), (Read(None),)])
    return backbone.forward(pack([job], torch.device('cuda:0')))[FINAL]


@torch.inference_mode()
def check_model(layer_types):
    torch.manual_seed(19)
    cfg = Qwen3_5MoeTextConfig(vocab_size=512, hidden_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
        moe_intermediate_size=64, shared_expert_intermediate_size=64,
        num_experts=4, num_experts_per_tok=2, layer_types=layer_types,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, max_position_embeddings=512,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000., 'partial_rotary_factor': .25,
                         'mrope_section': [2, 1, 1], 'mrope_interleaved': True})
    original = Qwen3_5MoeForCausalLM(cfg).eval()
    # from_pretrained keeps rotary buffers in float32 while casting the parameters.
    for parameter in original.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)
    bf16 = {k: v.contiguous().clone() for k, v in original.state_dict().items()}
    fused, split = dict(bf16), dict(bf16)
    for name, value in bf16.items():
        if not name.endswith(('.experts.gate_up_proj', '.experts.down_proj')):
            continue
        q, scale = quantize_experts(value)
        fused[name], fused[name + '_scale'] = q, scale
        del split[name]
        prefix, parameter = name.rsplit('.', 1)
        for e in range(q.shape[0]):
            projections = [('down_proj', slice(None))] if parameter == 'down_proj' else [
                ('gate_proj', slice(0, q.shape[1] // 2)), ('up_proj', slice(q.shape[1] // 2, None))]
            for projection, rows in projections:
                key = f'{prefix}.{e}.{projection}.weight'
                split[key] = q[e, rows].contiguous().clone()
                split[key + '_scale'] = scale[e, rows].contiguous().clone()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for layout, weights, is_fp8 in [('bf16', bf16, False), ('fused', fused, True), ('split', split, True)]:
            write(root / layout, cfg, weights, is_fp8)
        reference = Qwen35(copy.deepcopy(original).to('cuda:0'))
        loaded = Qwen35.load(str(root / 'bf16'), device='cuda:0')
        assert torch.equal(forward(reference), forward(loaded))
        print('PASS BF16 checkpoint load gives the original model outputs', flush=True)
        loaded_fused = Qwen35.load(str(root / 'fused'), device='cuda:0', quantize='fp8')
        loaded_split = Qwen35.load(str(root / 'split'), device='cuda:0', quantize='fp8')
        for loaded_fp8 in (loaded_fused, loaded_split):
            for layer in loaded_fp8.layers:
                assert layer.mlp.experts.gate_up_proj.is_meta
                assert layer.mlp.experts.down_proj.is_meta
            for name, parameter in loaded_fp8.causal_lm.named_parameters():
                if '.experts.' not in name:
                    assert torch.equal(parameter, bf16[name].to(parameter.device)), name
        for fw, sw in zip(loaded_fused._w, loaded_split._w):
            for key in ['experts', 'experts_down']:
                assert torch.equal(fw[key].view(torch.uint8), sw[key].view(torch.uint8))
                assert torch.equal(fw[key + '_scale'], sw[key + '_scale'])
        a, b = forward(loaded_fused), forward(loaded_split)
        assert torch.isfinite(a).all() and torch.equal(a, b)
        print('PASS fused/split FP8 checkpoints give identical weights, scales and GPU outputs', flush=True)
        online = Qwen35.load(str(root / 'bf16'), device='cuda:0', quantize='fp8')
        assert torch.equal(forward(online), a)
        print('PASS existing BF16-to-FP8 quantization agrees with the equivalent FP8 checkpoint', flush=True)


def main():
    for layer_types in (['full_attention'] * 2, ['linear_attention', 'full_attention']):
        print('Checking', layer_types, flush=True)
        check_model(layer_types)


if __name__ == '__main__':
    main()
