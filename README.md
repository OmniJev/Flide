<p align="center">
  <img alt="Flide: Accelerating Decision Inference with Exact State Forking" src="assets/banner.svg" width="100%">
</p>

<p align="center"><b>English</b> · <a href="assets/README_zh.md">简体中文</a> · <a href="assets/README_ja.md">日本語</a></p>

<p align="center">
  <a href="https://pypi.org/project/flide/"><img alt="PyPI" src="https://img.shields.io/pypi/v/flide?style=flat-square&logo=pypi&logoColor=white&color=3A86FF"></a>
  <a href="https://github.com/OmniJev/Flide"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-Flide-18202F?style=flat-square&logo=github&logoColor=white"></a>
  <a href="https://flide.org"><img alt="Website" src="https://img.shields.io/badge/Website-flide.org-00A890?style=flat-square&logo=googlechrome&logoColor=white"></a>
  <a href="https://docs.flide.org"><img alt="Docs" src="https://img.shields.io/badge/Docs-latest-1F5FD1?style=flat-square&logo=materialformkdocs&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-EF476F?style=flat-square&logo=apache&logoColor=white"></a>
  <a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-F28E2B?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.12%2B-5B6680?style=flat-square&logo=pytorch&logoColor=white"></a>
</p>

<p align="center"><a href="#-news">News</a> · <a href="#speed">Speed</a> · <a href="#exact-answers">Exact answers</a> · <a href="#quick-start">Quick start</a> · <a href="#models">Models</a> · <a href="#options">Options</a></p>

[Flide](https://docs.flide.org) serves decision models. A request holds a state, such as an agent's trajectory or a document, and typed
questions about it. Flide forks the state into all of its questions inside one forward pass, so the state is read
once and every question gets the answer it would get if it were asked alone. Answers come back as probabilities
through the `/v1/systemone` API that vLLM and SGLang also serve.


## 🔥 News

- 🔥 **2026-10-25** We added two decision model families in Flide 0.1.2 (Intern-Decision, torchcast) and up to 256 options per question. PRs welcome.
- 🔥 **2026-10-24** We released Flide 0.1.0 on [PyPI](https://pypi.org/project/flide/). Welcome to have a try.


<p align="center">
  <img alt="Fork the state into every question: a generation engine runs every question as a sequence and computes part of the state again, multi-item scoring adds delimiters that change the answers, and Flide forks the state at its last token" src="assets/fork.svg" width="100%">
</p>

With three questions, a generation engine runs three sequences and computes 4 cells of the state again, and multi-item scoring adds 3 delimiters that change the answers. Flide runs one sequence, computes nothing twice, adds no tokens, and gives each question the answer it gets when asked alone.

## Speed

<p align="center">
  <img alt="Speed of Flide, vLLM and SGLang on one H200 with Qwen3.5-0.8B" src="assets/speed.svg" width="100%">
</p>

On one H200 with Qwen3.5-0.8B, Flide is up to 9.4 times faster than both vLLM and SGLang (QuALITY, where it is 14 times faster than vLLM), 3.6 to 4 times faster on RACE and the Decision Index, and serves 2.4 times the throughput with 64 clients. The more questions a state carries, the bigger the gap.

## Exact answers

<p align="center">
  <img alt="Shared work, exact answers: Flide stays within 1.1 points of a float32 reference, multi-item scoring moves 0.9 to 14.8 points; with --batch-invariant a request gets the same bits however it is batched" src="assets/exact.svg" width="100%">
</p>

Flide stays within 1.1 points of a float32 reference, the closest of any engine, and flips at most 2.3% of decisions at 0.5. Multi-item scoring moves answers by up to 14.8 points and flips up to 28.9%. With `--batch-invariant`, a request gets the same bits however it is batched.

## Quick start

See the [docs](https://docs.flide.org) for every model, the API and all options.

### 1. Install

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install flide torchvision
```

### 2. Start a server

Any chat model on Qwen3.5 or Gemma 4 serves as it is and answers through its next token.

```bash
flide serve --model Qwen/Qwen3.5-4B
```

OneJev models read their answers their own way, and Flide serves them as they are too.

```bash
flide serve --model OmniJev/OneJev1.6-0.8B
```

An Open-Jev release is an adapter on a Qwen3.5 base. Merge it into a checkpoint once, then serve that.

```bash
python -m flide_open_jev.prepare ZefanCai/Open-Jev-2B open-jev-2b
flide serve --model open-jev-2b
```

`flide serve` listens on all interfaces at port 8000; pass `--host 127.0.0.1` to keep it local.

### 3. Send a request

```bash
curl http://localhost:8000/v1/systemone -H "Content-Type: application/json" -d '{
  "state": "Order #1182 arrived with a cracked screen. The customer asks for a refund and attached two photos.",
  "questions": {
    "action": {"type": "choice", "instructions": "What should support do next?",
               "criteria": {"refund": "refund the order", "replace": "send a replacement", "escalate": "ask a human"}},
    "urgent": {"type": "noul", "instructions": "Does this need a reply within the hour?"}
  }
}'
```

OneJev models also read images. Put `<image:1>` where the photo belongs in the state and send the photo in `media`.

```bash
IMG=$(base64 -w0 photo.jpg)
curl http://localhost:8000/v1/systemone -H "Content-Type: application/json" -d @- <<EOF
{
  "state": {"ticket": "Order #1182 arrived with a cracked screen. The customer asks for a refund.", "photo": "<image:1>"},
  "media": [{"type": "image", "data": "data:image/jpeg;base64,$IMG"}],
  "questions": {
    "cracked": {"type": "noul", "instructions": "The photo shows a cracked screen"},
    "action": {"type": "choice", "instructions": "What should support do next?",
               "criteria": {"refund": "refund the order", "replace": "send a replacement", "escalate": "ask a human"}}
  }
}
EOF
```

Every answer carries the probability of each option, ready for a threshold.

## Models

| Model | Checkpoints | Inputs |
|---|---|---|
| [Gemma 4](https://huggingface.co/collections/google/gemma-4-69ce8ad93186d46744cb42f1) | `google/gemma-4-E2B-it`, `-E4B-it`, `-12B-it`, `-31B-it` | text, images |
| [Intern-Decision](https://huggingface.co/collections/internlm/intern-decision-6ab75776af19ca84df0bc004) | `internlm/Intern-Decision-4B`, `-2B`, `-0.8B` | text |
| [OneJev](https://huggingface.co/collections/OmniJev/onejev16-6ad27669cb868abe0516486d) | `OmniJev/OneJev1.6-0.8B`, `-4B`, `-9B`, `-27B` | text, images, video |
| [Open-Jev](https://huggingface.co/collections/ZefanCai/open-jev-6ab049b9d43a267bae4dedc8) | `ZefanCai/Open-Jev-2B`, `-9B` | text |
| [Qwen3.5](https://huggingface.co/collections/Qwen/qwen35-6992e3053c019221cf2d725f) | `Qwen/Qwen3.5-0.8B`, `-2B`, `-4B`, `-9B`, `-27B` | text, images |
| [torchcast](https://huggingface.co/torchcast-ai/torchcast-decision-27b) | `torchcast-ai/torchcast-decision-27b` | text |

Any chat model on Qwen3.5 or Gemma 4 serves as it is; their rows name the official ones. The other rows are
decision models, each read the way its release reads it. A new family is one folder in `plugins/` and one line
in `pyproject.toml`; a separate package can also join through the `flide.formats` entry point.

## Options

| Option | What it does |
|---|---|
| `--session-cache-gb 40` | keeps the states of earlier requests, so an agent's next step reads only what is new |
| `--batch-invariant` | the same bits for a request whatever it is batched with, at some cost in speed |
| `--quantize fp8` | Qwen3.5 in FP8 on Ada, Hopper and Blackwell GPUs |
| `--frontends 2` | HTTP, rendering and tokenizing in two processes of their own, for many clients |

## License

Apache-2.0. `flide/kernels/gdn.py` holds a kernel derived from two kernels of
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (MIT); THIRD_PARTY_NOTICES lists it with
the other code Flide builds on.
