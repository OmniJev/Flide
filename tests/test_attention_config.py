"""The attention tile config, without a GPU.

    python tests/test_attention_config.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.kernels import attention


def config(smem, capability, dtype, head_dim):
    real_props, real_cap = torch.cuda.get_device_properties, torch.cuda.get_device_capability
    torch.cuda.get_device_properties = lambda device: SimpleNamespace(shared_memory_per_block_optin=smem)
    torch.cuda.get_device_capability = lambda device: capability
    try:
        return attention.config("cuda:0", dtype, head_dim)
    finally:
        torch.cuda.get_device_properties, torch.cuda.get_device_capability = real_props, real_cap


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

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()