# Optional attention backends

`flide serve` and `flide run` accept `--attention-backend auto|triton|cute`.
Python callers can pass `Engine(..., attention_backend="cute")`. The default is
`FLIDE_ATTENTION_BACKEND`, or `triton` when unset. Select it before model loading and
CUDA Graph capture; selection is process-wide, like `batch_invariant`. Do not
change it on an existing engine or mix engines with different selections in one
process.

- `auto`: use the optional CuTe backend for supported inputs; missing dependencies
  retain Triton.
- `triton`: always use the existing implementation, without importing CuTe.
- `cute`: require the optional runtime at startup, and use it for supported inputs.
  Unsupported attention semantics still use Triton. CUDA execution errors propagate.

## Qualification status

CuTe is opt-in. Kernel equivalence tests pass within BF16 tolerances, but this
is not a guarantee of identical full-model answers. The 2026-11-17 fused
KnowLine-35B FP8 check on the fixed 140-question / approximately 34k-token fixture
found a maximum probability change of 0.071822, a mean absolute change of
0.017343, and seven crossings of 0.5 compared with Triton. These are numerical
comparisons, not a labeled accuracy evaluation. Retain the default Triton backend
until the downstream application accepts these differences. Explicitly selecting
`cute` opts into the alternative floating-point reduction order.

## Coverage

The fused configuration targets SM120 (including RTX PRO 6000 Blackwell), BF16
Q/K/V with matching head dimensions of 64, 128 or 256. It uses 64-query by 64-key tiles, 128 threads and
one pipeline stage. Configuration and compilation caches include the actual head
dimension; the validated set is used by the unified exact-range backend. Other dimensions retain Triton. This is an explicit support list, not
a claim to accept every dimension. FP8 model weights are supported when attention activations
are BF16; the attention calculation itself is not FP8.

The backend reuses FLIDE's existing query tiles and exact per-row ranges into
both KV sources. A tile may contain tokens from multiple questions. The kernel
visits the union of their key ranges, applies each row's own visibility intervals,
and skips masking on complete blocks visible to every row. Both sources share
one online softmax and output accumulator; no KV copy or output merge is needed.
The sources can have different storage, offsets and strides, including a session
context buffer. This also handles bidirectional prefixes, isolated groups and
windows already represented by exact clipped ranges.

`prepare_attention` builds reusable device metadata once per batch or graph,
validates range bounds against the source capacities, and marks covered rows.
The metadata kernels execute again on graph replay; no lengths are read back to
the CPU. Current producers cover Qwen3.5 packed and flat/session graph paths.
Other backbones retain their implementation until they supply exact metadata.
There is one CuTe execution path; causal prefixes are handled by the same
exact-range kernel as branches and other representable visibility patterns.
No 4096-token minimum is imposed.

Unsupported hardware, dtype/head dimensions, unaligned layouts, batch-invariant
mode, explicit window calls and final-layer subset queries retain Triton. Masks
that cannot be expressed by the two exact intervals must also retain Triton;
this is not a claim to support arbitrary sparse masks. Invalid/oversized tiles
are not marked covered. CUDA execution errors propagate rather than silently
falling back after a failed launch.

The default remains Triton. `cute` explicitly opts into fusion for supported
inputs; this does not mean CuTe wins on every shape. In particular, metadata and
scheduler overhead can dominate tiny inputs. Performance qualification uses
separate kernel and full-model benchmarks, including numerical output comparison.

## Installation

See [optional CuTe dependencies](cute-installation.md) for installation, provenance
and migration from the earlier experimental runtime. Then select the backend:

```bash
flide serve --model /path/to/model --attention-backend cute
```

## Validation

```bash
python tests/test_attention_backends.py
python tests/test_cute_exact_attention.py
python tests/test_cute_head_dims.py
```

The tests cover exact dual-source visibility, bidirectional and clipped-window
ranges, isolated groups, zero-visible-key rows, dynamic graph replay, missing
optional dependencies, mixed masks and branches, strided V,
multiple sequences, empty graph slots, context index mappings, resumed prefixes,
unsupported-input fallback, and CUDA Graph replay with changed metadata. GPU checks
skip when SM120 or the optional runtime is unavailable. CuTe changes floating-point
reduction order; compare model probabilities as well as kernel tolerances when
qualifying another model. Use `--attention-backend triton` for a comparable baseline.

Head-dimension checks cover 64/128/256, interleaved compilation, MHA/GQA,
strided dual-source KV, empty visibility, CUDA Graph replay and pure causal inputs.
Small inputs are also checked against an independent dense float64 reference.
New head dimensions have kernel-level qualification; full-model qualification
must still be performed for the intended model.
