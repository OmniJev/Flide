"""Shared packed forward for Qwen2 and full-attention Qwen3 text checkpoints."""
from __future__ import annotations

import logging

import torch
from torch.nn import functional as F

from ... import lora, settings
from ...formats.base import FINAL
from ...kernels import attention
from ...kernels.fp8 import mm
from ...runtime.pack import PackedBatch
from ..base import Backbone
from ..load import blank, build, load_into
from .ops import ArenaContext, rms_norm, rotary, swiglu


def _merge(linears):
    """Combine projections that read the same input, keeping the modules' weights as views."""
    weight = torch.cat([linear.weight.detach() for linear in linears])
    for linear, rows in zip(linears, weight.split([linear.out_features for linear in linears])):
        linear.weight.data = rows
    return lora.attached(weight, linears)


def _norm(module, hidden, residual=None):
    return rms_norm(hidden, module.weight, module.variance_epsilon, residual=residual)


def _project(x, weight, bias):
    if bias is None:
        return mm(x, weight)
    if isinstance(weight, torch.Tensor) and not settings.batch_invariant:
        return F.linear(x, weight, bias)
    return mm(x, weight) + bias


class _Context:
    """The packed prefix stream plus the cached K/V rows before each resume point."""

    def __init__(self, batch):
        self.batch = batch
        self.meta = batch.ctx["meta"]

    def keys(self, layer, k, v):
        batch = self.batch
        ctx = batch.ctx
        prefix = batch.n_prefix_tokens
        keys = k.new_empty((ctx["len"], *k.shape[1:]))
        values = v.new_empty((ctx["len"], *v.shape[1:]))
        keys.index_copy_(0, ctx["index"], k[:prefix])
        values.index_copy_(0, ctx["index"], v[:prefix])
        for job, entry, start in batch.resumes:
            at = ctx["start"][job]
            old_k, old_v = entry.kv[layer]
            keys[at:at + start].copy_(old_k[:start])
            values[at:at + start].copy_(old_v[:start])
        for job, marks in batch.marks.items():
            mark = max(marks)
            at = ctx["start"][job]
            save = batch.saved.setdefault(job, {"states": {mark: {}}, "kv": {}})["kv"]
            key_rows, value_rows = keys[at:at + mark], values[at:at + mark]
            if isinstance(save, dict):
                save[layer] = (key_rows.clone(), value_rows.clone())
            else:
                save.write(layer, key_rows, value_rows)
        return keys, values


def _arena_context(batch):
    """Reuse the session graph's fused gather and save in packed forwards with arena-backed entries."""
    import numpy as np

    from ...runtime.session import ArenaRows

    resumes = [(job, start, entry.kv) for job, entry, start in batch.resumes]
    saves = [(job, max(marks), batch.saved.get(job, {}).get("kv")) for job, marks in batch.marks.items()]
    records = resumes + saves
    if not records or any(not isinstance(rows, ArenaRows) for _, _, rows in records):
        return _Context(batch)
    store = records[0][2].arena
    if any(rows.arena is not store for _, _, rows in records):
        return _Context(batch)

    ctx = batch.ctx
    indices = np.full((2, ctx["len"]), -1, dtype=np.int32)
    for job, start, rows in resumes:
        at = ctx["start"][job]
        indices[0, at:at + start] = -2 - rows.rows["full"][:start]
    for job, mark, rows in saves:
        at = ctx["start"][job]
        indices[1, at:at + mark] = rows.rows["full"]
    source, destination = torch.from_numpy(indices).to(store.device, non_blocking=True)
    new = torch.arange(batch.n_prefix_tokens, dtype=torch.int32, device=store.device)
    source.index_copy_(0, ctx["index"], new)
    return ArenaContext(store, source, destination)


class Qwen2(Backbone):
    model_types = ("qwen2",)
    block_attention = True
    prefix_attention = True
    session_states = True

    @staticmethod
    def _architecture(config):
        from transformers import Qwen2ForCausalLM, Qwen2Model

        return Qwen2Model if "Qwen2Model" in (config.architectures or ()) else Qwen2ForCausalLM

    @staticmethod
    def _mlp_weights(mlp, tensor_parallel):
        from ...runtime.tp import RowParallel

        down = lora.attached(mlp.down_proj.weight, [mlp.down_proj])
        return _merge((mlp.gate_proj, mlp.up_proj)), RowParallel(down) if tensor_parallel else down

    @staticmethod
    def _ffn(x, weights):
        gate_up, down = weights
        return mm(swiglu(mm(x, gate_up)), down)

    @classmethod
    def load(cls, model_dir: str, device: str = "cuda:0", dtype: str = "bfloat16", loader: str = "direct",
             merge_lora: bool = False, cuda_graphs: bool = False, quantize: str | None = None, tp=None) -> "Qwen2":
        """Load a text checkpoint, with CUDA graphs when requested by the engine."""
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        name = config.model_type
        if name not in cls.model_types:
            raise ValueError(f"{cls.__name__}: found model_type {name!r}, expected one of {cls.model_types}")
        if quantize is not None:
            raise NotImplementedError(f"{name}: --quantize is not supported")
        device = torch.device(device)
        dtype = getattr(torch, dtype)
        if device.type != "cuda":
            raise ValueError(f"{name}: packed inference requires CUDA")
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f"{name}: expected dtype bfloat16, float16 or float32")
        if getattr(config, "quantization_config", None) is not None:
            raise NotImplementedError(f"{name}: quantized checkpoints are not supported")
        if getattr(config, "dual_chunk_attention_config", None) is not None:
            raise NotImplementedError(f"{name}: dual-chunk attention is not supported")
        layers = getattr(config, "layer_types", None)
        full_attention = config.sliding_window is None if layers is None else set(layers) == {"full_attention"}
        if not full_attention or config.hidden_act != "silu":
            raise NotImplementedError(f"{name}: expected full attention and SwiGLU")
        if loader not in ("direct", "hf", "none"):
            raise ValueError(f"{name}: unknown loader {loader!r}")

        arch = cls._architecture(config)
        if tp is not None and tp[1] > 1:
            if loader != "direct":
                raise ValueError(f"{name}: tensor parallel requires the direct safetensors loader")
            from .tp import read_shard

            return cls(read_shard(arch, config, model_dir, device, dtype, tp, merge_lora),
                       cuda_graphs=cuda_graphs, tensor_parallel=True)
        model = None
        if loader == "none":
            model = build(arch, config, device, dtype)
            blank(model)
        elif loader == "direct":
            try:
                model = build(arch, config, device, dtype)
                load_into(model, model_dir, device, merge=merge_lora)
            except (ValueError, FileNotFoundError) as exc:
                logging.getLogger("flide.load").warning("direct load failed (%s); using transformers", exc)
                model = None
        if model is None:
            model = arch.from_pretrained(model_dir, config=config, dtype=dtype, device_map={"": str(device)},
                                         local_files_only=True).eval()
        return cls(model, cuda_graphs=cuda_graphs)

    def __init__(self, model, cuda_graphs: bool = False, tensor_parallel: bool = False):
        from ...runtime.tp import RowParallel

        self.model = model
        self.lm = getattr(model, "model", model)
        self.device, self.dtype = model.device, model.dtype
        self.num_layers = len(self.lm.layers)
        self.config = model.config
        fixed_rope = self.lm.rotary_emb.rope_type in ("default", "linear", "yarn")
        self.cuda_graphs = cuda_graphs and fixed_rope
        self.session_states = fixed_rope and not tensor_parallel
        self._graphs = None
        first = self.lm.layers[0].self_attn
        self.head_dim = first.head_dim
        self.qkv_widths = first.q_proj.out_features, first.k_proj.out_features, first.v_proj.out_features
        self.attn_cfg = attention.config(self.device, self.dtype, self.head_dim)
        if torch.cuda.get_device_capability(self.device) == (8, 0) and self.dtype != torch.float32 and self.head_dim <= 128:
            # A100 keeps more thread blocks resident with a smaller query tile than Hopper.
            self.attn_cfg = {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2}
        self.projections = []
        for layer in self.lm.layers:
            attn, mlp = layer.self_attn, layer.mlp
            qkv = (attn.q_proj, attn.k_proj, attn.v_proj)
            bias = torch.cat([projection.bias.detach() for projection in qkv]) if qkv[0].bias is not None else None
            out = lora.attached(attn.o_proj.weight, [attn.o_proj])
            if tensor_parallel:
                out = RowParallel(out)
            self.projections.append((_merge(qkv), bias, out, self._mlp_weights(mlp, tensor_parallel),
                                     getattr(attn, "q_norm", None), getattr(attn, "k_norm", None)))
        last_qkv, last_bias, *_ = self.projections[-1]
        q_width = self.qkv_widths[0]
        self.last_qkv = (last_qkv[:q_width], last_qkv[q_width:]) if isinstance(last_qkv, torch.Tensor) else last_qkv.split(q_width)
        self.last_bias = (None, None) if last_bias is None else (last_bias[:q_width], last_bias[q_width:])

    def request_graphs(self):
        if self.cuda_graphs and self._graphs is None:
            from .graphs import Qwen2Graphs

            self._graphs = Qwen2Graphs(self)
        return self._graphs

    def session_rows(self, mark: int) -> dict:
        return {"full": mark}

    def kv_arena(self, nbytes: int):
        from ...runtime.session import KVArena

        row = (self.config.num_key_value_heads, self.head_dim)
        layers = {i: ("full", row, row) for i in range(self.num_layers)}
        return KVArena(layers, nbytes, self.device, self.dtype, {"full": 1.0})

    def output_head(self):
        head = self.model.get_output_embeddings()
        if head is None:
            raise ValueError("this checkpoint has no output head")
        return lora.output_head(head)

    def _tiles(self, ranges, groups=0):
        cfg = self.attn_cfg
        return attention.block_ranges(*ranges, cfg["BLOCK_M"], cfg["BLOCK_N"], exact=True, groups=groups)

    @torch.inference_mode()
    def forward(self, batch: PackedBatch, pad_to: int | None = None) -> dict:
        if batch.vision or batch.pos3 is not None or batch.set_ranges is not None:
            raise ValueError(f"{self.config.model_type}: expected a text-only batch with standard positions")
        hidden = batch.embeds if batch.embeds is not None else self.lm.embed_tokens(batch.input_ids)
        meta = (batch.job.int(), batch.branch.int(), batch.positions.int())
        ctx = _arena_context(batch) if batch.ctx is not None else None
        return self.decode(hidden, batch.positions, meta, batch.key_ranges.T, batch.attn_runs, batch.reads, ctx)

    def decode(self, hidden, positions, meta, ranges, groups, reads, ctx=None):
        ints = [layer for layer in reads if layer is not FINAL]
        if any(layer < 0 or layer > self.num_layers for layer in ints):
            raise ValueError(f"{self.config.model_type}: read layer must be between 0 and {self.num_layers}, or FINAL")
        outputs = {0: hidden[reads[0]]} if 0 in reads else {}
        top = self.num_layers if FINAL in reads else max(ints, default=0)
        if top == 0:
            return outputs

        cos, sin = self.lm.rotary_emb(hidden, positions[None, :])
        cos, sin = cos[0], sin[0]
        tiles = self._tiles(ranges, groups)
        key_meta = meta if ctx is None else ctx.meta
        q_width, k_width, v_width = self.qkv_widths
        q_heads, k_heads = q_width // self.head_dim, k_width // self.head_dim
        # The final readout only needs the attention and MLP outputs at its selected rows.
        final_rows = reads[FINAL] if FINAL in reads and self.num_layers not in reads else None
        if final_rows is not None:
            read_ranges = ranges.index_select(1, final_rows)
            read_tiles = self._tiles(read_ranges, final_rows.numel())
            read_meta = tuple(column[final_rows] for column in meta)

        x = _norm(self.lm.layers[0].input_layernorm, hidden)
        for i in range(top):
            layer = self.lm.layers[i]
            qkv, bias, out, mlp, q_norm, k_norm = self.projections[i]
            if i + 1 == self.num_layers and final_rows is not None:
                q_weight, kv_weight = self.last_qkv
                q_bias, kv_bias = self.last_bias
                q = _project(x.index_select(0, final_rows), q_weight, q_bias)
                k, v = _project(x, kv_weight, kv_bias).chunk(2, dim=-1)
                q = rotary(q.unflatten(-1, (q_heads, self.head_dim)), cos[final_rows], sin[final_rows], q_norm)
                k = rotary(k.unflatten(-1, (k_heads, self.head_dim)), cos, sin, k_norm)
                q_ranges, q_tiles, q_meta = read_ranges, read_tiles, read_meta
                hidden = hidden[final_rows]
            else:
                qk, v = _project(x, qkv, bias).split((q_width + k_width, v_width), dim=-1)
                rotated = rotary(qk.unflatten(-1, (q_heads + k_heads, self.head_dim)), cos, sin, q_norm, k_norm, q_heads)
                q, k = rotated.split((q_heads, k_heads), dim=1)
                q_ranges, q_tiles, q_meta = ranges, tiles, meta
            v = v.unflatten(-1, (k_heads, self.head_dim))
            k1, v1 = (k, v) if ctx is None else ctx.keys(i, k, v)
            attended = attention.attention(q, k1, v1, k, v, q_meta, key_meta, meta, q_tiles,
                                           layer.self_attn.scaling, cfg=self.attn_cfg, qranges=q_ranges)
            attended = _project(attended.flatten(1), out, layer.self_attn.o_proj.bias)
            hidden, x = _norm(layer.post_attention_layernorm, hidden, attended)
            update = self._ffn(x, mlp)
            if i + 1 < top:
                hidden, x = _norm(self.lm.layers[i + 1].input_layernorm, hidden, update)
            else:
                hidden = hidden + update
            if i + 1 in reads:
                outputs[i + 1] = hidden[reads[i + 1]]

        if FINAL in reads:
            outputs[FINAL] = _norm(self.lm.norm, hidden if final_rows is not None else hidden[reads[FINAL]])
        return outputs
