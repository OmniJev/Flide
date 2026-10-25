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

[Flide](https://docs.flide.org) 是决策模型的推理引擎。一个请求里有一个状态，比如智能体的轨迹或一篇文档，还有关于它的若干个问题。
Flide 在一次前向传播里把状态分叉给所有问题，状态只读一遍，每个问题拿到的答案和单独问它时完全一样。
答案以概率的形式返回，接口是 vLLM 和 SGLang 也在用的 `/v1/systemone`。


## 🔥 动态

- 🔥 **2026-10-25** Flide 0.1.2 新增两个决策模型家族（Intern-Decision、torchcast），每道题最多支持 256 个选项。欢迎提 PR。
- 🔥 **2026-10-24** Flide 0.1.0 已发布到 [PyPI](https://pypi.org/project/flide/)，欢迎试用。


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

[文档](https://docs.flide.org)里有每个模型家族、接口、选项和 Python 引擎的说明。

## 模型

| 模型 | 检查点 | 输入 |
|---|---|---|
| [Gemma 4](https://huggingface.co/collections/google/gemma-4-69ce8ad93186d46744cb42f1) | `google/gemma-4-E2B-it`, `-E4B-it`, `-12B-it`, `-31B-it` | 文本、图片 |
| [Intern-Decision](https://huggingface.co/collections/internlm/intern-decision-6ab75776af19ca84df0bc004) | `internlm/Intern-Decision-4B`, `-2B`, `-0.8B` | 文本 |
| [OneJev](https://huggingface.co/collections/OmniJev/onejev16-6ad27669cb868abe0516486d) | `OmniJev/OneJev1.6-0.8B`, `-4B`, `-9B`, `-27B` | 文本、图片、视频 |
| [Open-Jev](https://huggingface.co/collections/ZefanCai/open-jev-6ab049b9d43a267bae4dedc8) | `ZefanCai/Open-Jev-2B`, `-9B` | 文本 |
| [Qwen3.5](https://huggingface.co/collections/Qwen/qwen35-6992e3053c019221cf2d725f) | `Qwen/Qwen3.5-0.8B`, `-2B`, `-4B`, `-9B`, `-27B` | 文本、图片 |
| [torchcast](https://huggingface.co/torchcast-ai/torchcast-decision-27b) | `torchcast-ai/torchcast-decision-27b` | 文本 |

基于 Qwen3.5 或 Gemma 4 的聊天模型都能直接使用，表中列的是官方版本。其余几行是决策模型，各自按其发布版的方式读答案。
新增一个模型族，只需在 `plugins/` 里加一个目录，再在 `pyproject.toml` 里登记一行；单独发布的包也可以通过
`flide.formats` 入口点接入。

## 选项

| 选项 | 作用 |
|---|---|
| `--session-cache-gb 40` | 保存之前请求的状态，智能体的下一步只读新增的部分 |
| `--batch-invariant` | 无论和哪些请求同批，同一个请求的结果逐位相同，速度略慢 |
| `--quantize fp8` | 在 Ada、Hopper、Blackwell 显卡上以 FP8 运行 Qwen3.5 |
| `--frontends 2` | HTTP、渲染和分词放到两个独立进程里，适合大量客户端 |

## 许可证

Apache-2.0。`flide/kernels/gdn.py` 中有一个内核改写自
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention)（MIT）的两个内核，
THIRD_PARTY_NOTICES 列出了它和 Flide 用到的其他代码。
