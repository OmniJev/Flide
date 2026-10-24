"""Turn an Open-Jev release (a LoRA adapter, a scalar head and a temperature on a pinned Qwen3.5 base) into a
checkpoint directory flide loads directly: the base text decoder with the adapter merged in (W + alpha / r
* B @ A, computed in float32), the head as head.safetensors, and decision_config.json naming the format.

    python -m flide_open_jev.prepare ZefanCai/Open-Jev-2B /path/to/out
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

FORMAT_ID = "open-jev-candidate-v1"


def prepare(repo: str, out: str, revision: str | None = None) -> Path:
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = Path(repo) if Path(repo).is_dir() else Path(snapshot_download(repo, revision=revision))
    ck = src / "package" / "checkpoint" if (src / "package" / "checkpoint").is_dir() else src
    model = json.loads((ck / "model.json").read_text())
    temperature = json.loads((ck / "temperature.json").read_text())["temperature"]
    base = Path(snapshot_download(model["model_id"], revision=model["revision"],
                                  allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "merges.txt"]))
    adapter_cfg = json.loads((ck / "adapter" / "adapter_config.json").read_text())
    scale = adapter_cfg["lora_alpha"] / adapter_cfg["r"]
    lora = {}
    with safe_open(str(ck / "adapter" / "adapter_model.safetensors"), "pt") as f:
        for k in f.keys():
            lora[k] = f.get_tensor(k)
    deltas = {}
    for k in lora:
        if ".lora_A." not in k:
            continue
        name = k.split("base_model.model.", 1)[1].replace(".lora_A.weight", ".weight")
        A, B = lora[k].float(), lora[k.replace(".lora_A.", ".lora_B.")].float()
        deltas[name] = scale * (B @ A)

    config = json.loads((base / "config.json").read_text())
    text = dict(config.get("text_config", config))
    text["architectures"] = ["Qwen3_5ForCausalLM"]
    text["model_type"] = "qwen3_5_text"
    text["tie_word_embeddings"] = config.get("tie_word_embeddings", text.get("tie_word_embeddings", False))
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights, used = {}, set()
    files = sorted(base.glob("*.safetensors"))
    for fp in files:
        with safe_open(str(fp), "pt") as f:
            for k in f.keys():
                if not k.startswith("model.language_model."):
                    continue
                name = k[len("model.language_model."):]
                t = f.get_tensor(k)
                if name in deltas:
                    t = (t.float() + deltas[name]).to(t.dtype)
                    used.add(name)
                weights["model." + name] = t.contiguous()
    missing = sorted(set(deltas) - used)
    if missing:
        raise ValueError(f"adapter weights with no base tensor: {missing[:5]}")
    save_file(weights, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    head = torch.load(ck / "head.pt", map_location="cpu", weights_only=True)
    save_file({"weight": head["weight"].float().contiguous(), "bias": head["bias"].float().contiguous()},
              str(out_dir / "head.safetensors"))
    (out_dir / "config.json").write_text(json.dumps(text, indent=2))
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt",
                 "special_tokens_map.json", "generation_config.json"):
        if (base / name).exists():
            shutil.copy(base / name, out_dir / name)
    (out_dir / "decision_config.json").write_text(json.dumps({
        "format_id": FORMAT_ID, "package": "flide-open-jev", "source": repo, "base": model["model_id"],
        "base_revision": model["revision"], "temperature": temperature, "max_length": model["max_length"],
        "lora": {"r": adapter_cfg["r"], "alpha": adapter_cfg["lora_alpha"], "merged": len(deltas)}}, indent=2))
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("out")
    ap.add_argument("--revision", default=None)
    a = ap.parse_args()
    print(prepare(a.repo, a.out, a.revision))


if __name__ == "__main__":
    main()
