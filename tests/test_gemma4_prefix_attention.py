"""Gemma 4 under attention "prefix" without a GPU: the global layers read each prefix both ways and each branch one way
over its prefix and itself; the sliding layers stay causal inside their window. Checked against a reference mask for
the packed path (Gemma4.attention_meta) and the CUDA-graph staging (GemmaGraphs._stage then _body).

    python tests/test_gemma4_prefix_attention.py

The attention kernel reads, per query tile, the union of the queries' key ranges: whole key blocks inside every range
unmasked, the rest filtered by the metadata (job, branch, position, block number, window). When, for every query, the
keys in its ranges are exactly the keys the metadata shows, the result does not depend on the tiling; both must equal
the reference mask. Block attention in the same round keeps its own mask.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from flide.backbones import get as backbone_for  # noqa: E402
from flide.backbones.gemma4 import Gemma4  # noqa: E402
from flide.backbones.gemma4.graphs import GemmaGraphs  # noqa: E402
from flide.formats.base import Read  # noqa: E402
from flide.runtime.pack import Job, pack  # noqa: E402

W = 4   # a small window, so its edge is covered
# (prefix, branches): one branch longer than the window, and a tree of one branch (the graphs run it as one sequence)
TREES = [([1, 2, 3, 4, 5, 6], [[10, 11], [12], [13, 14, 15]]),
         ([7, 8, 9], [[20, 21, 22]]),
         ([30, 31, 32, 33, 34], [[40], [41, 42, 43, 44, 45, 46]])]
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} {name}{'' if ok else f' ({detail})'}")


def reference(ident, layer, modes, window=W):
    """[T, T]: does query q see key k. ident[i] = (tree, branch or -1, index) or None for padding; modes[tree] is
    "prefix" or "blocks"."""
    T = len(ident)
    out = torch.zeros(T, T, dtype=torch.bool)
    for q, a in enumerate(ident):
        if a is None:
            out[q, q] = True
            continue
        for k, b in enumerate(ident):
            if b is None or a[0] != b[0] or (b[1] >= 0 and b[1] != a[1]):
                continue
            p = len(TREES[a[0]][0])
            qp = a[2] if a[1] < 0 else p + a[2]
            kp = b[2] if b[1] < 0 else p + b[2]
            if layer == "full_attention" or not window:   # no window: every layer as the global ones
                if modes[a[0]] == "blocks":   # every block whole: the prefix, and a branch with its prefix
                    out[q, k] = (a[1] < 0 and b[1] < 0) or a[1] >= 0
                else:
                    out[q, k] = (a[1] < 0 and b[1] < 0) or (a[1] >= 0 and (b[1] < 0 or kp <= qp))
            else:
                out[q, k] = kp <= qp and (not window or qp - kp < window)
    return out


def visible(meta, window):
    """The kernel's two tests per (query, key): inside the query's key ranges, and shown by the metadata."""
    job, br, pos = (x.long() for x in meta.tok[:3])
    blk = meta.tok[3].long() if len(meta.tok) > 3 else None
    k = torch.arange(job.shape[0])[None, :]
    r = meta.ranges.long()
    in_range = ((k >= r[:, 0:1]) & (k < r[:, 1:2])) | ((k >= r[:, 2:3]) & (k < r[:, 3:4]))
    seen = pos[None, :] <= pos[:, None]
    if blk is not None:
        seen = seen | ((blk[None, :] == blk[:, None]) & (blk[:, None] >= 0))
    vis = (job[None, :] == job[:, None]) & ((br[None, :] == -1) | (br[None, :] == br[:, None])) & seen
    if window:
        vis = vis & ((pos[:, None] - pos[None, :]) < window)
    return in_range, vis


def compare(name, metas, ident, modes, window=W):
    for layer, m in metas.items():
        in_range, vis = visible(m, window if layer == "sliding_attention" else 0)
        check(f"{name} {layer}: key ranges and metadata agree", torch.equal(in_range, vis),
              int((in_range != vis).sum()))
        want = reference(ident, layer, modes, window)
        check(f"{name} {layer}: the reference mask", torch.equal(vis, want), int((vis != want).sum()))


def packed(modes, no_window=False):
    jobs = [Job(list(p), [list(s) for s in ss], [(Read(None),)] * len(ss), blocks=mode == "blocks",
                prefix_two_way=mode == "prefix") for (p, ss), mode in zip(TREES, modes)]
    b = pack(jobs, "cpu")
    ident = [(t, -1, i) for t, (p, _) in enumerate(TREES) for i in range(len(p))]
    ident += [(t, bi, i) for t, (_, ss) in enumerate(TREES) for bi, s in enumerate(ss) for i in range(len(s))]
    me = SimpleNamespace(config=SimpleNamespace(sliding_window=0 if no_window else W), bidir=False, placeholders=set(),
                         no_window=no_window)
    me._block_meta = lambda *a: Gemma4._block_meta(me, *a)
    tok = (b.job.int(), b.branch.int(), b.positions.int())
    return Gemma4.attention_meta(me, tok, b.key_ranges, b.input_ids.numpy(), b), ident


class _Stream:
    def synchronize(self):
        pass

    def record(self):
        pass


def staged(modes, L: int = 64, no_window=False, image=False):
    """GemmaGraphs._stage and _body on stand-ins for the device buffers, `_decode` catching the two metas. `image`: the
    first tree's prefix holds an image (placeholder tokens, on a model whose images read both ways)."""
    g = SimpleNamespace(bmax=64, staged=_Stream())
    g.bb = SimpleNamespace(bidir=image, placeholders={999}, pad_id=0, no_window=no_window,
                           config=SimpleNamespace(sliding_window=0 if no_window else W),
                           lm=SimpleNamespace(embed_tokens=lambda ids: torch.zeros(ids.shape[0], 1)),
                           encode_vision=lambda raw, emb, media, graphs=None: emb, vision_graphs=lambda: None)
    g.ids, g.raw, g.pos = (torch.zeros(L, dtype=torch.long) for _ in range(3))
    g.meta = torch.zeros(4, L, dtype=torch.int32)
    g.ranges = torch.zeros(L, 4, dtype=torch.int32)
    g.gblk = torch.zeros(L, dtype=torch.int32)
    g.reads = torch.zeros(64, dtype=torch.long)
    g.emb = torch.zeros(L, 1)
    g.host = torch.zeros(12, L, dtype=torch.long)
    g.host_reads = torch.zeros(64, dtype=torch.long)
    g.nreq = GemmaGraphs.nreq
    g.nread = lambda n: GemmaGraphs.nread(g, n)
    got = {}
    g._decode = lambda n, full, sliding: got.update(full_attention=full, sliding_attention=sliding)
    reqs = [(list(p), [list(s) for s in ss], None, "prefix" if mode == "prefix" else True)
            for (p, ss), mode in zip(TREES, modes)]
    if image:
        p0 = reqs[0][0]
        reqs[0] = (p0[:1] + [999] * (len(p0) - 2) + p0[-1:], reqs[0][1], ("image",), reqs[0][3])
    GemmaGraphs._stage(g, reqs, L)
    GemmaGraphs._body(g, L, True)
    ident = []
    for t, (p, ss) in enumerate(TREES):   # a one-branch tree runs as one sequence, the others' branches after them
        ident += [(t, -1, i) for i in range(len(p))]
        if len(ss) == 1:
            ident += [(t, 0, i) for i in range(len(ss[0]))]
    ident += [(t, bi, i) for t, (_, ss) in enumerate(TREES) if len(ss) > 1 for bi, s in enumerate(ss)
              for i in range(len(s))]
    return got, ident + [None] * (L - len(ident))


def main():
    check("Gemma 4 and its graphs run attention prefix", Gemma4.prefix_attention and GemmaGraphs.prefix_attention)
    check("gemma4_unified_text is a Gemma 4 model type",
          "gemma4_unified_text" in Gemma4.model_types and backbone_for("gemma4_unified_text") is Gemma4)
    for modes in (("prefix",) * 3, ("prefix", "blocks", "prefix")):
        tag = "+".join(dict.fromkeys(modes))
        metas, ident = packed(modes)
        compare(f"packed {tag}", metas, ident, modes)
        metas, ident = staged(modes)
        compare(f"graphs {tag}", metas, ident, modes)
    for modes in (("blocks",) * 3, ("blocks", "prefix", "blocks")):   # no_window: the sliding layers read every block whole
        tag = "+".join(dict.fromkeys(modes)) + " without a window"
        metas, ident = packed(modes, True)
        compare(f"packed {tag}", metas, ident, modes, 0)
        metas, ident = staged(modes, no_window=True)
        compare(f"graphs {tag}", metas, ident, modes, 0)
    modes = ("blocks",) * 3   # an image in a prefix, no window: every layer reads its block whole, the image's tokens too
    metas, ident = staged(modes, no_window=True, image=True)
    compare("graphs blocks with an image without a window", metas, ident, modes, 0)
    print("ALL PASS" if fails == 0 else f"{fails} FAILED")
    return fails


if __name__ == "__main__":
    sys.exit(main())
