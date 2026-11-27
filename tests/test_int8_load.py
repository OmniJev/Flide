"""int8 loading (qwen3_5, no GPU needed): the decoder projections read as int8 (never held in 16 bits), then smoothed by
calibrate_int8 from each weight read again from the checkpoint, give the same int8 weights, scales, smoothing factors
and 16-bit tail rows, bit for bit, as a model read in 16 bits and quantized from the same input maxima; with
`--no-weights` (zeros) too."""
import tempfile
import weakref

import torch
import transformers

from flide.backbones.qwen35 import Qwen35
from flide.kernels import gpu, int8

gpu.int8 = lambda device=None: True   # the CPU stands in for the GPU: these products are never run here
KEYS = ("gate_up", "down", "in", "out", "qkv", "o")
_CKPT = None


def checkpoint():
    global _CKPT
    if _CKPT is None:
        cfg = transformers.Qwen3_5TextConfig(
            vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=4, num_attention_heads=8,
            num_key_value_heads=4, head_dim=32, linear_num_key_heads=4, linear_num_value_heads=8, linear_key_head_dim=32,
            linear_value_head_dim=32, layer_types=["linear_attention"] * 3 + ["full_attention"], tie_word_embeddings=False)
        torch.manual_seed(0)
        model = transformers.Qwen3_5ForCausalLM(cfg).to(torch.bfloat16)
        with torch.no_grad():
            for p in model.parameters():
                p.normal_(0, 0.05)
            model.model.layers[0].mlp.down_proj.weight[:, 7] *= 40   # an outlier channel
        d = tempfile.mkdtemp()
        model.save_pretrained(d)
        _CKPT = d
    return _CKPT


def maxima(bb) -> dict:
    """Input maxima for every product, a few channels never seen (0: left unsmoothed)."""
    g = torch.Generator().manual_seed(1)
    out = {}
    for i, w in enumerate(bb._w):
        for k in KEYS:
            if k in w:
                m = torch.rand(w[k].shape[1], generator=g) * 20
                m[::17] = 0
                out[(i, k)] = m
    return out


def calibrate(bb, stats) -> int:
    def run():   # as the calibration requests would leave them
        for i, w in enumerate(bb._w):
            for k in KEYS:
                if k in w:
                    w[k].amax.copy_(stats[(i, k)])
    return bb.calibrate_int8(run)


def weights(bb) -> list:
    out = []
    for w in bb._w:
        for k in KEYS:
            if k in w:
                x = w[k]
                out.append((k, x.q, x.scale, x.inv, x.tail))
    return out


def same(a, b) -> bool:
    eq = lambda x, y: (x is None and y is None) or (x is not None and y is not None and torch.equal(x, y))
    return len(a) == len(b) and all(ka == kb and all(eq(x, y) for x, y in zip(ra, rb)) for (ka, *ra), (kb, *rb) in zip(a, b))


def check(loader):
    d = checkpoint()
    lean = Qwen35.load(d, device="cpu", quantize="int8", loader=loader)
    full = Qwen35.load(d, device="cpu", loader=loader)
    assert lean.int8 and not full.int8
    for layer in lean.layers:   # the int8 projections were never read in 16 bits
        assert layer.mlp.gate_proj.weight.is_meta and layer.mlp.down_proj.weight.is_meta
    g = full._w[0]["gate_up"]
    q, s = int8.quantize_rows(g)
    assert torch.equal(lean._w[0]["gate_up"].q, q) and torch.equal(lean._w[0]["gate_up"].scale, s.reshape(-1))
    assert lean._w[0]["gate_up"].inv is None   # unsmoothed until calibrated
    stats = maxima(full)
    n = calibrate(lean, stats)
    assert n == calibrate(full, stats) == len(stats)
    assert same(weights(lean), weights(full)), "int8 read lean and smoothed differs from int8 from 16 bits"
    assert all(x[3] is not None for x in weights(lean))   # every product smoothed
    assert lean._int8_reread is None and not lean._int8_names


def test_one_at_a_time():
    """While the products are smoothed, the int8 weights already replaced are gone (the load peak)."""
    bb = Qwen35.load(checkpoint(), device="cpu", quantize="int8", loader="direct")
    old = [weakref.ref(w[k].q) for w in bb._w for k in KEYS if k in w]
    alive, reread = [], bb._int8_reread

    def counted(names):
        alive.append(sum(r() is not None for r in old))
        return reread(names)

    bb._int8_reread = counted
    n = calibrate(bb, maxima(bb))
    assert n == len(old) == len(alive)
    assert all(a <= n - j + 1 for j, a in enumerate(alive)), alive


def test_products_in_16_bits():
    """The calibration's products are those of the 16-bit model: each weight read again is its 16-bit weight bit for bit, and
    an InputMax keeps the same maxima and gives the same product as one over the 16-bit weight, whole and in two halves."""
    lean = Qwen35.load(checkpoint(), device="cpu", quantize="int8", loader="direct")
    full = Qwen35.load(checkpoint(), device="cpu", loader="direct")
    g = torch.Generator().manual_seed(2)
    for i, (w, v) in enumerate(zip(lean._w, full._w)):
        for k in KEYS:
            if k not in w:
                continue
            read = lean._weight16(i, k, w[k])
            assert torch.equal(read(), v[k]), (i, k)
            x = torch.randn(5, v[k].shape[1], generator=g).to(torch.bfloat16)
            a, b = int8.InputMax(w[k], read), int8.InputMax(v[k])
            assert torch.equal(a(x), b(x)) and torch.equal(a.amax, b.amax) and float(a.amax.max()) > 0, (i, k)
            if w[k].tail is None:
                n = v[k].shape[0] // 2
                for p, q in zip(a.split(n), b.split(n)):
                    assert torch.equal(p(x), q(x)), (i, k)
                assert torch.equal(a.amax, b.amax)


def test_read():
    check("direct")


def test_no_weights():
    check("none")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
