"""The forward Gated DeltaNet kernels of flash-linear-attention 0.6.0 (MIT license, THIRD_PARTY_NOTICES)
that Qwen3.5's linear-attention layers run: the chunked gated delta rule, its causal convolution, L2 norm, beta
sigmoid and gate. The module layout is fla's, so autotuner picks recorded under fla's names still apply
(runtime/kcache.py). Imported only when a model has such layers."""
