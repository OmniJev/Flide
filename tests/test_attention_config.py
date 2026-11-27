"""The attention tile config, without a GPU.

    python tests/test_attention_config.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.kernels import attention, gpu


def config(smem, capability, dtype, head_dim, amd=False):
    """The tiles a part with this shared memory and compute capability gets; `amd` makes it an AMD part (whatever
    the torch build this test runs on)."""
    real_props, real_cap, real_amd = torch.cuda.get_device_properties, torch.cuda.get_device_capability, gpu.AMD
    torch.cuda.get_device_properties = lambda device: SimpleNamespace(shared_memory_per_block_optin=smem)
    torch.cuda.get_device_capability = lambda device: capability
    gpu.AMD = amd
    try:
        return attention.config("cuda:0", dtype, head_dim)
    finally:
        torch.cuda.get_device_properties, torch.cuda.get_device_capability, gpu.AMD = real_props, real_cap, real_amd


def main():
    fails = 0

    def check(name, got, want):
        nonlocal fails
        ok = all(got[k] == v for k, v in want.items())
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: {got}")

    # 256-dim heads on a 99 KB card: 32-query tiles with four warps on Blackwell, the older tile elsewhere
    check("blackwell, 99 KB, bf16, 256-dim gets four warps on the same tile", config(99 * 1024, (12, 0), torch.bfloat16, 256),
          {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2})
    check("Hopper, 99 KB, bf16, 256-dim keeps 64-query tiles", config(99 * 1024, (9, 0), torch.bfloat16, 256),
          {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 8, "num_stages": 1})
    check("blackwell, 227 KB, bf16, 256-dim keeps the 128-query tile", config(227 * 1024, (12, 0), torch.bfloat16, 256),
          {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 8, "num_stages": 2})
    check("blackwell, 99 KB, bf16, 128-dim is untouched", config(99 * 1024, (12, 0), torch.bfloat16, 128),
          {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 8, "num_stages": 2})
    check("float32 keeps its own tile", config(99 * 1024, (12, 0), torch.float32, 256),
          {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 4, "num_stages": 1})
    # AMD (MI210: 64 KB of LDS, reported as compute capability 9.0): the tiles that pass test_attention there
    mi210 = (64 * 1024, (9, 0))
    check("MI210, bf16, 128-dim", config(*mi210, torch.bfloat16, 128, amd=True),
          {"BLOCK_M": 128, "BLOCK_N": 32, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2})
    check("MI210, bf16, 256-dim", config(*mi210, torch.bfloat16, 256, amd=True),
          {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2})
    check("MI210, bf16, 512-dim", config(*mi210, torch.bfloat16, 512, amd=True),
          {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 4, "num_stages": 1})
    check("MI210, float32 keeps its own tile", config(*mi210, torch.float32, 128, amd=True),
          {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 4, "num_stages": 1})

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()