"""The Qwen3.5 vision tower for one request at a time, from CUDA graphs.

At batch size one the tower is bound by kernel launches (a 280-patch screenshot runs
almost all of its eager time in launches). Everything that depends only on the image grids (interpolated position
embeddings, rotary cos / sin, the attention segments) is computed once per set of grids with transformers'
own helpers; the patch embedding, the blocks and the merger are captured as one graph per set of grids and
replayed. An agent works at a fixed screen size, so one capture serves a whole episode; the least recently
used graphs are dropped beyond `max_graphs`.

The blocks read the checkpoint's own weights through fused steps (kernels/vision.py): the patch embedding as one
matrix product, LayerNorm together with the residual add before it, the queries' and keys' rotary embedding in one
kernel, attention from flide's kernel (every token of an image or video frame sees the others, no copies to
heads-first layouts); about 10 kernels a block where transformers runs about 25. The merger is the checkpoint's
module.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F

from ...kernels import attention as attn_kernel
from ...kernels import vision as vk
from ...runtime.tp import capture_mode


def patchify(img: torch.Tensor, patch: int, merge: int, temporal: int) -> torch.Tensor:
    """[3, H, W] -> [H * W / patch^2, 3 * temporal * patch^2]: the image processor's patch layout (Qwen2VL `patchify`,
    the same reshape, permute and temporal repeat), on whatever device the image is."""
    C, H, W = img.shape
    gh, gw = H // patch, W // patch
    x = img.reshape(1, C, gh // merge, merge, patch, gw // merge, merge, patch).permute(0, 2, 5, 3, 6, 1, 4, 7)
    return x.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, temporal, -1, -1).reshape(gh * gw, C * temporal * patch * patch)


def mrope_positions(types, image_grid, video_grid, merge: int):
    """M-RoPE positions [3, P] of a prefix and the offset of the token after it (position P + offset), as the
    checkpoint's `get_rope_index` gives them for one sequence, from its token types (0 text, 1 image, 2 video): runs
    of text count on from the current position; each run of vision tokens takes the next grid (videos one frame
    each) at (frame, row, column) of the merged grid plus the current position, which then moves on by the merged
    grid's longer side. Vectorized: transformers walks the tokens in Python (about 0.8 ms a screenshot request)."""
    import numpy as np

    types = np.asarray(types)
    n = len(types)
    frames = []
    if video_grid is not None:
        frames = [(1, int(h), int(w)) for t, h, w in video_grid.tolist() for _ in range(int(t))]
    grids = {1: iter(image_grid.tolist() if image_grid is not None else []), 2: iter(frames)}
    cuts = np.flatnonzero(np.diff(types)) + 1
    starts, ends = np.concatenate([[0], cuts]), np.concatenate([cuts, [n]])
    pos = np.empty((3, n), np.int64)
    cur = 0
    for a, e in zip(starts.tolist(), ends.tolist()):
        kind = int(types[a])
        if kind == 0:
            pos[:, a:e] = np.arange(cur, cur + e - a)
            cur += e - a
            continue
        t, h, w = next(grids[kind])
        gh, gw = h // merge, w // merge
        if e - a != t * gh * gw:
            raise ValueError(f"{e - a} vision tokens for a grid of {t * gh * gw}")
        tt, hh, ww = np.meshgrid(np.arange(t), np.arange(gh), np.arange(gw), indexing="ij")
        pos[:, a:e] = np.stack([tt.ravel(), hh.ravel(), ww.ravel()]) + cur
        cur += max(h, w) // merge
    return torch.from_numpy(pos), int(pos.max()) + 1 - n if n else 0


def resized(img: torch.Tensor, size) -> torch.Tensor:
    """[3, h, w] uint8 at `size` (height, width), as the image processor resizes it (torch's antialiased bicubic for
    uint8): on the GPU by kernels/vision.resize_u8, bit for bit the same pixels, else torch's own CPU kernel."""
    if size is None or tuple(size) == tuple(img.shape[-2:]):
        return img
    if img.is_cuda:
        return vk.resize_u8(img, *size)
    return F.interpolate(img[None], size=tuple(size), mode="bicubic", align_corners=False, antialias=True)[0]


def pixels(v: dict, pix: str, device) -> torch.Tensor:
    """A request's pixel patches on `device`, float32 as the image processor gives them. They may come normalized
    (`pix`), as the processor's patches before rescaling and normalization (`pix`_u8: uint8, a quarter of the bytes),
    or for images as uint8 [3, H, W] each before the processor's patch reshuffle (`pixel_images_u8`, half the bytes of
    the patches, which repeat every pixel `temporal` times), resized here first when `pixel_resize` gives the sizes
    (then they come as decoded), reshuffled here; uint8 inputs are normalized as the processor does it (float32,
    subtract the per-column mean, divide by the standard deviation, `pixel_scale`), so the values are the same bits."""
    imgs = v.get("pixel_images_u8") if pix == "pixel_values" else None
    if imgs is not None:
        sizes = v.get("pixel_resize") or [None] * len(imgs)
        up = (lambda x: x.pin_memory().to(device, non_blocking=True)) if torch.device(device).type == "cuda" else \
            (lambda x: x.to(device))
        u8 = torch.cat([patchify(resized(up(x), s), *v["pixel_patch"]) for x, s in zip(imgs, sizes)])
    else:
        u8 = v.get(pix + "_u8")
        if u8 is None:
            return v[pix].to(device)
        u8 = u8.to(device, non_blocking=True)
    mean, std = (t.to(device, non_blocking=True) for t in v["pixel_scale"])
    return (u8.float() - mean) / std

def _segment_attention(q, k, v, segs, scale: float) -> torch.Tensor:
    """[n, H, D] each; every token sees the tokens of its own segment. PyTorch's attention, cuDNN first."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    order = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    heads = lambda x: x.transpose(0, 1)[None]
    with sdpa_kernel(order, set_priority=True):
        if len(segs) == 1:
            return F.scaled_dot_product_attention(heads(q), heads(k), heads(v), scale=scale)[0].transpose(0, 1)
        o = torch.empty_like(q)
        for a, e in segs:
            o[a:e] = F.scaled_dot_product_attention(heads(q[a:e]), heads(k[a:e]), heads(v[a:e]),
                                                    scale=scale)[0].transpose(0, 1)
        return o


class VisionGraphs:
    def __init__(self, vlm, max_graphs: int = 8):
        self.visual = vlm.visual
        self.config = self.visual.config
        self.max_graphs = max_graphs
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: OrderedDict = OrderedDict()

    def _grid_constants(self, grid: torch.Tensor):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as M

        v = self.visual
        idx, w = M.get_vision_interpolation_indices_and_weights(
            grid, num_grid_per_side=v.num_grid_per_side, mode=v.interpolation_mode,
            align_corners=v.interpolation_align_corners, spatial_merge_size=self.config.spatial_merge_size, kwargs={})
        pos_ids = M.get_vision_position_ids(grid, v.spatial_merge_size, kwargs={})
        pos = (v.pos_embed(idx) * w[:, :, None]).sum(1)
        dummy = torch.zeros(1, device=pos.device, dtype=v.patch_embed.proj.weight.dtype)
        cos, sin = v.rotary_pos_emb(dummy, pos_ids)
        seg = (grid[:, 1] * grid[:, 2]).repeat_interleave(grid[:, 0]).to(pos.device)
        n = int(seg.sum())
        job = torch.repeat_interleave(torch.arange(seg.shape[0], device=pos.device), seg)
        ends = torch.cumsum(seg, 0)
        lo = (ends - seg)[job]
        hi = ends[job]
        meta = (job.int(), torch.full((n,), -1, dtype=torch.int32, device=pos.device),
                torch.zeros(n, dtype=torch.int32, device=pos.device))
        cfg = attn_kernel.config(pos.device, dummy.dtype, v.blocks[0].attn.head_dim)
        zero = torch.zeros_like(lo)
        R = attn_kernel.block_ranges(lo, hi, zero, zero, cfg["BLOCK_M"], cfg["BLOCK_N"], exact=True)
        segs = None
        if dummy.dtype in (torch.bfloat16, torch.float16):
            e = torch.cumsum(seg, 0).tolist()
            segs = list(zip([0] + e[:-1], e))
        return pos, cos.float().contiguous(), sin.float().contiguous(), meta, R, cfg, segs

    def _body(self, pixels, consts):
        pos, cos, sin, meta, R, cfg, segs = consts
        v = self.visual
        pe = v.patch_embed.proj
        n = pixels.shape[0]
        x = F.linear(pixels.reshape(n, -1).to(pe.weight.dtype), pe.weight.reshape(pe.weight.shape[0], -1), pe.bias)
        blocks = v.blocks
        h, x = vk.layer_norm(x, blocks[0].norm1.weight, blocks[0].norm1.bias, blocks[0].norm1.eps,
                             residual=pos.to(x.dtype))
        for bi, blk in enumerate(blocks):
            a = blk.attn
            qkv = F.linear(x, a.qkv.weight, a.qkv.bias).view(n, 3, a.num_heads, -1)
            q, k = vk.rope(qkv, cos, sin)
            if segs is None:
                o = attn_kernel.attention(q, k, qkv[:, 2], k, qkv[:, 2], meta, meta, meta, R, a.scaling, 0, cfg)
            else:
                o = _segment_attention(q, k, qkv[:, 2], segs, a.scaling)
            d = F.linear(o.view(n, -1), a.proj.weight, a.proj.bias)
            h, x = vk.layer_norm(h, blk.norm2.weight, blk.norm2.bias, blk.norm2.eps, residual=d)
            mlp = blk.mlp
            y = F.gelu(F.linear(x, mlp.linear_fc1.weight, mlp.linear_fc1.bias), approximate="tanh")
            d = F.linear(y, mlp.linear_fc2.weight, mlp.linear_fc2.bias)
            if bi + 1 < len(blocks):
                nxt = blocks[bi + 1].norm1
                h, x = vk.layer_norm(h, nxt.weight, nxt.bias, nxt.eps, residual=d)
            else:
                h = h + d
        return v.merger(h)

    @torch.inference_mode()
    def __call__(self, pixels: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Merged vision features [tokens, hidden] for the images (or video) of one request (grids best on the host:
        the graph is looked up by them without waiting for the device)."""
        key = (tuple(map(tuple, grid.tolist())), tuple(pixels.shape), pixels.dtype)
        entry = self.graphs.get(key)
        if entry is None:
            consts = self._grid_constants(grid.to(pixels.device))
            buf = torch.empty_like(pixels)
            buf.copy_(pixels)
            for _ in range(2):
                self._body(buf, consts)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool, capture_error_mode=capture_mode()):
                out = self._body(buf, consts)
            torch.cuda.synchronize()
            entry = self.graphs[key] = (g, buf, out, consts)
            while len(self.graphs) > self.max_graphs:
                self.graphs.popitem(last=False)
        else:
            self.graphs.move_to_end(key)
        g, buf, out, *_ = entry
        buf.copy_(pixels)
        g.replay()
        return out.clone()
