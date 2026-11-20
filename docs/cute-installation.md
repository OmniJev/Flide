# Optional CuTe dependencies

From a Flide checkout, with the intended Python environment active:

```bash
python -m pip install '.[cute]'
```

For an editable development install use `python -m pip install -e '.[cute]'`.
The `cute` extra declares CUTLASS DSL 4.8.1 with CUDA 13 support and the pinned
TVM FFI, DLPack and Quack dependencies. Pip resolves the official DSL library
packages, including `nvidia-cutlass-dsl-libs-core`, which provides the Python DSL
sources. Do not use `--no-deps` for the first install. No CUTLASS checkout, source
copying, external runtime directory or custom `.pth` file is required.

FlashAttention CuTe is vendored under `flide/kernels/vendor/flash_attn/cute` and
shipped in both the wheel and source distribution. It is pinned to upstream
revision `328dd7b5a6905c75500a9bce334ac012763409a9`; `UPSTREAM.json` records original
file hashes and local changes. Imports are namespaced under Flide to avoid
collisions with another installed FlashAttention. Kernel code and numerical
configuration are unchanged. The BSD license and contributor list ship with the
package and are recorded in `THIRD_PARTY_NOTICES`. Maintainers should update the
source, provenance, licenses and validation results together.

A normal `pip install .` keeps CuTe dependencies optional. Installing this extra
alone does not change Flide inference behavior. This configuration targets Linux, Python 3.12 and
CUDA 13; other environments need validation.

### Migrating the earlier experimental installation

Remove the old environment's `flide_optional_cute.pth` before installing the extra:

```bash
python -c "from pathlib import Path; import sysconfig; (Path(sysconfig.get_path('purelib')) / 'flide_optional_cute.pth').unlink(missing_ok=True)"
python -m pip install '.[cute]'
```

The earlier `scripts/install-cute.sh` has been removed. Old experiment/runtime
folders are no longer required by Flide; keep them only if needed for reproducing
those experiments.

## Verify installation

```bash
python -c "import cutlass; from flide.kernels.vendor.flash_attn.cute import flash_attn_varlen_func; print('CuTe dependencies available')"
```

This checks the packaged kernel imports and official DSL dependencies. It does
not enable a model backend or run GPU inference.
