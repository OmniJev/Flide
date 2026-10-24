"""The Gemma 4 vision tower from CUDA graphs, one per input shape (an image processor pads every image to the same
number of patches, so one graph serves an agent's whole episode).

The graph runs transformers' own modules from the patch embedding through the encoder, the pooler, the
standardization and the projection into the language model (every step after the pooler is per token); only the
removal of the padding tokens, whose count depends on the image, runs after the replay. The least recently used
graphs are dropped beyond `max_graphs`.
"""
from __future__ import annotations

from collections import OrderedDict

import torch


class GemmaVisionGraphs:
    def __init__(self, vlm, max_graphs: int = 8):
        self.inner = vlm.model
        self.tower = self.inner.vision_tower
        self.max_graphs = max_graphs
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: OrderedDict = OrderedDict()

    def _body(self, pixels, positions):
        vt = self.tower
        k = vt.config.pooling_kernel_size
        padding = (positions == -1).all(dim=-1)
        x = vt.patch_embedder(pixels, positions, padding)
        out = vt.encoder(inputs_embeds=x, attention_mask=~padding, pixel_position_ids=positions)
        h, keep = vt.pooler(hidden_states=out.last_hidden_state, pixel_position_ids=positions, padding_positions=padding,
                            output_length=pixels.shape[-2] // (k * k))
        if vt.config.standardize:
            h = (h - vt.std_bias.float()) * vt.std_scale.float()
        return self.inner.embed_vision(inputs_embeds=h.to(x.dtype)), keep

    @torch.inference_mode()
    def __call__(self, pixels: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Projected vision features [tokens, hidden] of the images (padding tokens removed), in order."""
        key = (tuple(pixels.shape), pixels.dtype, tuple(positions.shape))
        entry = self.graphs.get(key)
        if entry is None:
            pbuf, qbuf = torch.empty_like(pixels), torch.empty_like(positions)
            pbuf.copy_(pixels)
            qbuf.copy_(positions)
            for _ in range(2):
                self._body(pbuf, qbuf)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                out = self._body(pbuf, qbuf)
            torch.cuda.synchronize()
            entry = self.graphs[key] = (g, pbuf, qbuf, out)
            while len(self.graphs) > self.max_graphs:
                self.graphs.popitem(last=False)
        else:
            self.graphs.move_to_end(key)
        g, pbuf, qbuf, (feats, keep) = entry
        pbuf.copy_(pixels)
        qbuf.copy_(positions)
        g.replay()
        return feats[keep]
