<p align="center">
  <img alt="Flide：以精确的状态分叉加速决策推理" src="zh/banner.svg" width="100%">
</p>

<p align="center"><a href="../README.md">English</a> · <b>简体中文</b> · <a href="README_ja.md">日本語</a></p>

<p align="center">
  <a href="https://pypi.org/project/flide/"><img alt="PyPI" src="https://img.shields.io/pypi/v/flide?style=flat-square&logo=pypi&logoColor=white&color=3A86FF"></a>
  <a href="https://github.com/OmniJev/Flide"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-Flide-18202F?style=flat-square&logo=github&logoColor=white"></a>
  <a href="https://flide.org"><img alt="Website" src="https://img.shields.io/badge/Website-flide.org-00A890?style=flat-square&logo=googlechrome&logoColor=white"></a>
  <a href="https://docs.flide.org"><img alt="Docs" src="https://img.shields.io/badge/Docs-latest-1F5FD1?style=flat-square&logo=materialformkdocs&logoColor=white"></a>
  <a href="../LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-EF476F?style=flat-square&logo=apache&logoColor=white"></a>
  <a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-F28E2B?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.12%2B-5B6680?style=flat-square&logo=pytorch&logoColor=white"></a>
</p>

<p align="center"><a href="#-动态">动态</a> · <a href="#速度">速度</a> · <a href="#精确的答案">精确的答案</a> · <a href="#快速开始">快速开始</a> · <a href="#模型">模型</a> · <a href="#选项">选项</a></p>

[Flide](https://docs.flide.org) 是面向 Jev 类决策模型的高速推理基础设施。凭借精确状态分叉，它比 vLLM 和 SGLang 最高快 9.4 倍，每个答案都和模型单独回答这道题时完全一致。接口是 Jev 的 `/v1/systemone`。


## 🔥 动态

- 🔥 **2026-10-28** Flide 0.1.5 新增六个决策模型家族，包括 [Jeffan](https://huggingface.co/collections/tattrongvu/decision-model-6ad5c85a79bd246cc1c18c4a)、[JevAny](https://huggingface.co/collections/SimpleJev/jevany-adaptive-decision-systems-6abc8fd39266b1c17b11b4e6)、[Kev](https://huggingface.co/collections/jaredpalmer/kev-6aad9d0ea49f2589665e07cd)、[Lychee](https://huggingface.co/LycheeAI/Lychee-27B)、[pplx-decider](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2) 和 [StartLux](https://huggingface.co/collections/startlux-models/startlux-decision-v11-instant-6aca8158623e8e001724cc82)，[Lumma](https://huggingface.co/collections/FrontiersMind/lumma-decision-models-6ab6356434b3db6856ec7ae6) 和 [mammoth](https://huggingface.co/baselabs/mammoth-1) 也能读图片了。
- 🔥 **2026-10-26** Flide 0.1.4 新增五个决策模型家族，包括 [cerno](https://huggingface.co/apertixai/cerno-27b)、[JADE](https://huggingface.co/theunnecessarythings/JADE)、[Linne](https://huggingface.co/collections/PNPM244/decision-model-6ad363fb3351cecf1112a5a8)、[mammoth](https://huggingface.co/baselabs/mammoth-1) 和 [Sieve](https://huggingface.co/collections/sthanika-ai/sieve-6abb93a61dd495f69811c586)，并支持 [Quyet-1.0-Large](https://huggingface.co/chinhnc/Quyet-1.0-Large)。
- 🔥 **2026-10-26** Flide 0.1.3 新增八个决策模型家族，包括 [AJev](https://huggingface.co/andyzhang232/ajev-gemma4-12b-lora7)、[Bespoke-Nimble](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v3)、[Darwin](https://huggingface.co/collections/FINAL-Bench/ztc-models-jev-ecosystems-6aafa630afd69b39ead55aa8)、[jiwo](https://huggingface.co/eljiwo/jiwo-4b)、[Lumma](https://huggingface.co/collections/FrontiersMind/lumma-decision-models-6ab6356434b3db6856ec7ae6)、[mogan](https://huggingface.co/moganai/mogan-decision-31B-it)、[Quyet](https://huggingface.co/collections/chinhnc/quyet-10-6ac1b597a9a6a6806bb01fa6) 和 [Senova](https://huggingface.co/WangSenQWQ/Senova-4B)，LoRA 适配器也能直接服务。
- 🔥 **2026-10-25** Flide 0.1.2 新增两个决策模型家族，包括 [Intern-Decision](https://huggingface.co/collections/internlm/intern-decision-6ab75776af19ca84df0bc004) 和 [torchcast](https://huggingface.co/torchcast-ai/torchcast-decision-27b)，每道题最多支持 256 个选项。
- 🔥 **2026-10-24** Flide 0.1.0 已发布到 [PyPI](https://pypi.org/project/flide/)，支持 [Qwen3.5](https://huggingface.co/collections/Qwen/qwen35-6992e3053c019221cf2d725f) 和 [Gemma 4](https://huggingface.co/collections/google/gemma-4-69ce8ad93186d46744cb42f1) 对话模型，以及 [OneJev](https://huggingface.co/collections/OmniJev/onejev16-6ad27669cb868abe0516486d) 和 [Open-Jev](https://huggingface.co/collections/ZefanCai/open-jev-6ab049b9d43a267bae4dedc8) 决策模型，欢迎试用。


<p align="center">
  <img alt="把状态分叉给每个问题：生成引擎为每个问题单独跑一条序列并重算部分状态，多项打分插入分隔符改变了答案，Flide 在状态末尾分叉" src="fork.svg" width="100%">
</p>

以 3 个问题为例，生成引擎要跑 3 条序列，重算 4 格状态；多项打分插入 3 个分隔符，答案随之改变。Flide 只跑 1 条序列，什么都不重算，不加任何 token，每个问题的答案都和单独问它时一样。

## 速度

<p align="center">
  <img alt="Flide、vLLM 与 SGLang 在一块 H200 上的速度，Qwen3.5-0.8B" src="speed.svg" width="100%">
</p>

在一块 H200 上跑 Qwen3.5-0.8B，Flide 最高比 vLLM 和 SGLang 都快 9.4 倍（QuALITY，比 vLLM 快 14 倍），在 RACE 和 Decision Index 上快 3.6 到 4 倍，64 个客户端并发时吞吐是它们的 2.4 倍。一个状态带的问题越多，差距越大。

## 精确的答案

<p align="center">
  <img alt="共享计算，答案精确：Flide 离 float32 参考最多 1.1 个点，多项打分偏 0.9 到 14.8 个点；开启 --batch-invariant 后，无论怎么组批，同一个请求的比特都一样" src="exact.svg" width="100%">
</p>

Flide 离 float32 参考最多 1.1 个点，是所有引擎里最近的，在 0.5 处翻转的决策最多 2.3%。多项打分会把答案挪动最多 14.8 个点，翻转最多 28.9% 的决策。开启 `--batch-invariant` 后，无论怎么组批，同一个请求得到的比特都一样。

## 快速开始

模型、接口和全部选项见[文档](https://docs.flide.org)。

### 1. 安装

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install flide torchvision
```

### 2. 启动服务

Qwen3.5 和 Gemma 4 上的聊天模型都能直接启动，用下一个 token 作答。

```bash
flide serve --model Qwen/Qwen3.5-4B
```

OneJev 用自己的方式读答案，Flide 同样可以直接启动。

```bash
flide serve --model OmniJev/OneJev1.6-0.8B
```

Open-Jev 发布的是 Qwen3.5 底座上的适配器，先合并成一个检查点，再启动它。

```bash
python -m flide_open_jev.prepare ZefanCai/Open-Jev-2B open-jev-2b
flide serve --model open-jev-2b
```

`flide serve` 默认监听所有网卡的 8000 端口，只想本机访问时加 `--host 127.0.0.1`。

### 3. 发送请求

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

OneJev 模型也能读图片。在 state 里图片该出现的位置写 `<image:1>`，再把图片放进 `media` 一起发送。

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

每个答案都带着每个选项的概率，可以直接拿去和阈值比较。

## 模型

后三列是 Flide 回答单个请求比该模型发布版自带的引擎快多少倍，取每个数据集上请求的中位延迟，在一张 H200 上测，DI 指 Decision Index。聊天模型对比的是 vLLM 和 SGLang 中更快的那个，OneJev 对比的是它换用 Flide 之前的 qev，写成范围的是测过的几个尺寸。

| 模型 | 输入 | DI | MultiRC | QuALITY |
|---|---|---:|---:|---:|
| [AJev](https://huggingface.co/andyzhang232/ajev-gemma4-12b-lora7) | 文本 | 3.7× | 9.3× | 2.6× |
| [Bespoke-Nimble](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v3) | 文本 | 2.6× | 2.1× | 1.6× |
| [cerno](https://huggingface.co/apertixai/cerno-27b) | 文本 | 3.4 to 4.7× | 8.1 to 9.1× | 1.1× |
| [Darwin](https://huggingface.co/collections/FINAL-Bench/ztc-models-jev-ecosystems-6aafa630afd69b39ead55aa8) | 文本 | 1.6× | 7.0× | 1.2× |
| [Gemma 4](https://huggingface.co/collections/google/gemma-4-69ce8ad93186d46744cb42f1) | 文本、图片 | 1.1 to 1.5× | 1.1 to 2.5× | 1.0 to 1.1× |
| [Intern-Decision](https://huggingface.co/collections/internlm/intern-decision-6ab75776af19ca84df0bc004) | 文本、图片 | 4.4× | 1.1× | 1.5× |
| [JADE](https://huggingface.co/theunnecessarythings/JADE) | 文本 | 2.0× | 9.2× | 1.7× |
| [Jeffan](https://huggingface.co/collections/tattrongvu/decision-model-6ad5c85a79bd246cc1c18c4a) | 文本 | 3.1× | 2.6× | 1.9× |
| [JevAny](https://huggingface.co/collections/SimpleJev/jevany-adaptive-decision-systems-6abc8fd39266b1c17b11b4e6) | 文本、图片 | 1.5× | 5.8× | 1.2× |
| [jiwo](https://huggingface.co/eljiwo/jiwo-4b) | 文本 | 1.7 to 2.1× | 5.0 to 6.9× | 1.8× |
| [Kev](https://huggingface.co/collections/jaredpalmer/kev-6aad9d0ea49f2589665e07cd) | 文本 | 5.6 to 10.1× | 1.3 to 1.7× | 1.1 to 2.2× |
| [Linne](https://huggingface.co/collections/PNPM244/decision-model-6ad363fb3351cecf1112a5a8) | 文本 | 2.3× | 8.6× | 1.2× |
| [Lumma](https://huggingface.co/collections/FrontiersMind/lumma-decision-models-6ab6356434b3db6856ec7ae6) | 文本、图片 | 1.7× | 7.1× | 1.2× |
| [Lychee](https://huggingface.co/LycheeAI/Lychee-27B) | 文本 | 1.4× | 1.5× | 1.4× |
| [mammoth](https://huggingface.co/baselabs/mammoth-1) | 文本、图片 | 3.3 to 5.3× | 2.3 to 3.4× | 1.6 to 2.0× |
| [mogan](https://huggingface.co/moganai/mogan-decision-31B-it) | 文本 | 1.6× | 8.5× | 1.8× |
| [OneJev](https://huggingface.co/collections/OmniJev/onejev16-6ad27669cb868abe0516486d) | 文本、图片、视频 | 3.2 to 11.9× | 7.3× | 4.7× |
| [Open-Jev](https://huggingface.co/collections/ZefanCai/open-jev-6ab049b9d43a267bae4dedc8) | 文本 | 4.2× | 5.0× | 1.3 to 1.7× |
| [pplx-decider](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2) | 文本、图片 | 3.1× | 1.4× | 1.2× |
| [Quyet](https://huggingface.co/collections/chinhnc/quyet-10-6ac1b597a9a6a6806bb01fa6) | 文本 | 1.9 to 5.9× | 7.0 to 7.2× | 1.4 to 1.9× |
| [Qwen3.5](https://huggingface.co/collections/Qwen/qwen35-6992e3053c019221cf2d725f) | 文本、图片 | 1.2 to 3.3× | 2.4 to 7.7× | 1.0 to 2.0× |
| [Senova](https://huggingface.co/WangSenQWQ/Senova-4B) | 文本 | 3.1× | 28.1× | 1.2× |
| [Sieve](https://huggingface.co/collections/sthanika-ai/sieve-6abb93a61dd495f69811c586) | 文本 | 3.0× | 14.3× | 2.6× |
| [StartLux](https://huggingface.co/collections/startlux-models/startlux-decision-v11-instant-6aca8158623e8e001724cc82) | 文本 | 2.6× | 1.1× | 1.2× |
| [torchcast](https://huggingface.co/torchcast-ai/torchcast-decision-27b) | 文本、图片 | 1.6× | 8.4× | 1.4× |

## 选项

| 选项 | 作用 |
|---|---|
| `--session-cache-gb 40` | 保存之前请求的状态，智能体的下一步只读新增的部分 |
| `--batch-invariant` | 无论和哪些请求同批，同一个请求的结果逐位相同，速度略慢 |
| `--quantize fp8` | 在 Ada、Hopper、Blackwell 显卡上以 FP8 运行 Qwen3.5 |
| `--frontends 2` | HTTP、渲染和分词放到两个独立进程里，适合大量客户端 |

## 许可证

Apache-2.0。`flide/kernels/gdn.py` 中有一个内核改写自 [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)（MIT）的两个内核，THIRD_PARTY_NOTICES 列出了它和 Flide 用到的其他代码。
