<p align="center">
  <img alt="Flide：厳密な状態フォークによる意思決定推論の高速化" src="ja/banner.svg" width="100%">
</p>

<p align="center"><a href="../README.md">English</a> · <a href="README_zh.md">简体中文</a> · <b>日本語</b></p>

<p align="center">
  <a href="https://pypi.org/project/flide/"><img alt="PyPI" src="https://img.shields.io/pypi/v/flide?style=flat-square&logo=pypi&logoColor=white&color=3A86FF"></a>
  <a href="https://github.com/OmniJev/Flide"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-Flide-18202F?style=flat-square&logo=github&logoColor=white"></a>
  <a href="https://flide.org"><img alt="Website" src="https://img.shields.io/badge/Website-flide.org-00A890?style=flat-square&logo=googlechrome&logoColor=white"></a>
  <a href="https://docs.flide.org"><img alt="Docs" src="https://img.shields.io/badge/Docs-latest-1F5FD1?style=flat-square&logo=materialformkdocs&logoColor=white"></a>
  <a href="../LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-EF476F?style=flat-square&logo=apache&logoColor=white"></a>
  <a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-F28E2B?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.12%2B-5B6680?style=flat-square&logo=pytorch&logoColor=white"></a>
</p>

<p align="center"><a href="#-ニュース">ニュース</a> · <a href="#速度">速度</a> · <a href="#正確な答え">正確な答え</a> · <a href="#クイックスタート">クイックスタート</a> · <a href="#モデル">モデル</a> · <a href="#オプション">オプション</a></p>

[Flide](https://docs.flide.org) は Jev 型の意思決定モデルのための高速な推論基盤です。厳密な状態分岐により vLLM や SGLang より最大 9.4 倍速く、どの答えもモデルがその質問だけに答えたときと同じです。API は Jev の `/v1/systemone` です。


## 🔥 ニュース

- 🔥 **2026-10-28** Flide 0.1.5 で [Jeffan](https://huggingface.co/collections/tattrongvu/decision-model-6ad5c85a79bd246cc1c18c4a)、[JevAny](https://huggingface.co/collections/SimpleJev/jevany-adaptive-decision-systems-6abc8fd39266b1c17b11b4e6)、[Kev](https://huggingface.co/collections/jaredpalmer/kev-6aad9d0ea49f2589665e07cd)、[Lychee](https://huggingface.co/LycheeAI/Lychee-27B)、[pplx-decider](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2)、[StartLux](https://huggingface.co/collections/startlux-models/startlux-decision-v11-instant-6aca8158623e8e001724cc82) の 6 つの意思決定モデルファミリーを追加し、[Lumma](https://huggingface.co/collections/FrontiersMind/lumma-decision-models-6ab6356434b3db6856ec7ae6) と [mammoth](https://huggingface.co/baselabs/mammoth-1) で画像入力にも対応しました。
- 🔥 **2026-10-26** Flide 0.1.4 で [cerno](https://huggingface.co/apertixai/cerno-27b)、[JADE](https://huggingface.co/theunnecessarythings/JADE)、[Linne](https://huggingface.co/collections/PNPM244/decision-model-6ad363fb3351cecf1112a5a8)、[mammoth](https://huggingface.co/baselabs/mammoth-1)、[Sieve](https://huggingface.co/collections/sthanika-ai/sieve-6abb93a61dd495f69811c586) の 5 つの意思決定モデルファミリーを追加し、[Quyet-1.0-Large](https://huggingface.co/chinhnc/Quyet-1.0-Large) にも対応しました。
- 🔥 **2026-10-26** Flide 0.1.3 で [AJev](https://huggingface.co/andyzhang232/ajev-gemma4-12b-lora7)、[Bespoke-Nimble](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v3)、[Darwin](https://huggingface.co/collections/FINAL-Bench/ztc-models-jev-ecosystems-6aafa630afd69b39ead55aa8)、[jiwo](https://huggingface.co/eljiwo/jiwo-4b)、[Lumma](https://huggingface.co/collections/FrontiersMind/lumma-decision-models-6ab6356434b3db6856ec7ae6)、[mogan](https://huggingface.co/moganai/mogan-decision-31B-it)、[Quyet](https://huggingface.co/collections/chinhnc/quyet-10-6ac1b597a9a6a6806bb01fa6)、[Senova](https://huggingface.co/WangSenQWQ/Senova-4B) の 8 つの意思決定モデルファミリーを追加し、LoRA アダプターもそのまま提供できるようになりました。
- 🔥 **2026-10-25** Flide 0.1.2 で [Intern-Decision](https://huggingface.co/collections/internlm/intern-decision-6ab75776af19ca84df0bc004)、[torchcast](https://huggingface.co/torchcast-ai/torchcast-decision-27b) の 2 つの意思決定モデルファミリーを追加し、1 問あたり最大 256 個の選択肢に対応しました。
- 🔥 **2026-10-24** Flide 0.1.0 を [PyPI](https://pypi.org/project/flide/) で公開しました。[Qwen3.5](https://huggingface.co/collections/Qwen/qwen35-6992e3053c019221cf2d725f) と [Gemma 4](https://huggingface.co/collections/google/gemma-4-69ce8ad93186d46744cb42f1) のチャットモデル、[OneJev](https://huggingface.co/collections/OmniJev/onejev16-6ad27669cb868abe0516486d) と [Open-Jev](https://huggingface.co/collections/ZefanCai/open-jev-6ab049b9d43a267bae4dedc8) の意思決定モデルに対応しています。ぜひお試しください。


<p align="center">
  <img alt="状態をすべての質問に分岐する：生成エンジンは質問ごとにシーケンスを走らせて状態の一部を再計算し、マルチアイテム採点は区切り記号で答えを変え、Flide は状態の最後のトークンで分岐する" src="fork.svg" width="100%">
</p>

質問が 3 つの場合、生成エンジンは 3 本のシーケンスを走らせて状態の 4 マスを再計算し、マルチアイテム採点は区切り記号を 3 つ足して答えを変えてしまいます。Flide は 1 本のシーケンスで済み、何も再計算せず、トークンも足さず、どの質問も単独で聞いたときと同じ答えになります。

## 速度

<p align="center">
  <img alt="H200 1 枚、Qwen3.5-0.8B での Flide、vLLM、SGLang の速度" src="speed.svg" width="100%">
</p>

H200 1 枚で Qwen3.5-0.8B を動かすと、Flide は vLLM と SGLang のどちらよりも最大 9.4 倍速く（QuALITY、vLLM 比では 14 倍）、RACE と Decision Index では 3.6 から 4 倍速く、64 クライアントではスループットが 2.4 倍になります。1 つの状態が抱える質問が多いほど、差は広がります。

## 正確な答え

<p align="center">
  <img alt="計算を共有しても答えは正確：Flide は float32 の参照から 1.1 ポイント以内、マルチアイテム採点は 0.9 から 14.8 ポイントずれる。--batch-invariant でバッチに関係なく同じビット" src="exact.svg" width="100%">
</p>

Flide は float32 の参照から 1.1 ポイント以内に収まり、どのエンジンよりも近く、0.5 で反転する判定は最大 2.3% です。マルチアイテム採点は答えを最大 14.8 ポイント動かし、判定を最大 28.9% 反転させます。`--batch-invariant` を付けると、どうバッチされても同じリクエストは同じビットを返します。

## クイックスタート

モデル、API、すべてのオプションは[ドキュメント](https://docs.flide.org)を参照してください。

### 1. インストール

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install flide torchvision
```

### 2. サーバーを起動する

Qwen3.5 と Gemma 4 上のチャットモデルはそのまま起動でき、次のトークンで答えます。

```bash
flide serve --model Qwen/Qwen3.5-4B
```

独自の方法で答えを読む OneJev も、そのまま起動できます。

```bash
flide serve --model OmniJev/OneJev1.6-0.8B
```

Open-Jev は Qwen3.5 ベースのアダプタとして公開されています。一度チェックポイントにマージしてから起動します。

```bash
python -m flide_open_jev.prepare ZefanCai/Open-Jev-2B open-jev-2b
flide serve --model open-jev-2b
```

`flide serve` はすべてのインターフェースのポート 8000 で待ち受けます。ローカルだけにするには `--host 127.0.0.1` を付けてください。

### 3. リクエストを送る

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

OneJev モデルは画像も読めます。state の中で写真が入る位置に `<image:1>` と書き、写真を `media` に入れて一緒に送ります。

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

答えにはそれぞれの選択肢の確率が付くので、そのまましきい値と比べられます。

## モデル

最後の 3 列は、1 リクエストに答えるのに Flide がリリース付属のエンジンより何倍速いかを示し、各データセットのリクエストの中央値レイテンシを H200 1 枚で測っています。DI は Decision Index です。チャットモデルは vLLM と SGLang の速いほう、OneJev は Flide の前に使っていた qev と比べており、範囲は試したサイズの幅です。

| モデル | 入力 | DI | MultiRC | QuALITY |
|---|---|---:|---:|---:|
| [AJev](https://huggingface.co/andyzhang232/ajev-gemma4-12b-lora7) | テキスト | 3.7× | 9.3× | 2.6× |
| [Bespoke-Nimble](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v3) | テキスト | 2.6× | 2.1× | 1.6× |
| [cerno](https://huggingface.co/apertixai/cerno-27b) | テキスト | 3.4 to 4.7× | 8.1 to 9.1× | 1.1× |
| [Darwin](https://huggingface.co/collections/FINAL-Bench/ztc-models-jev-ecosystems-6aafa630afd69b39ead55aa8) | テキスト | 1.6× | 7.0× | 1.2× |
| [Gemma 4](https://huggingface.co/collections/google/gemma-4-69ce8ad93186d46744cb42f1) | テキスト、画像 | 1.1 to 1.5× | 1.1 to 2.5× | 1.0 to 1.1× |
| [Intern-Decision](https://huggingface.co/collections/internlm/intern-decision-6ab75776af19ca84df0bc004) | テキスト、画像 | 4.4× | 1.1× | 1.5× |
| [JADE](https://huggingface.co/theunnecessarythings/JADE) | テキスト | 2.0× | 9.2× | 1.7× |
| [Jeffan](https://huggingface.co/collections/tattrongvu/decision-model-6ad5c85a79bd246cc1c18c4a) | テキスト | 3.1× | 2.6× | 1.9× |
| [JevAny](https://huggingface.co/collections/SimpleJev/jevany-adaptive-decision-systems-6abc8fd39266b1c17b11b4e6) | テキスト、画像 | 1.5× | 5.8× | 1.2× |
| [jiwo](https://huggingface.co/eljiwo/jiwo-4b) | テキスト | 1.7 to 2.1× | 5.0 to 6.9× | 1.8× |
| [Kev](https://huggingface.co/collections/jaredpalmer/kev-6aad9d0ea49f2589665e07cd) | テキスト | 5.6 to 10.1× | 1.3 to 1.7× | 1.1 to 2.2× |
| [Linne](https://huggingface.co/collections/PNPM244/decision-model-6ad363fb3351cecf1112a5a8) | テキスト | 2.3× | 8.6× | 1.2× |
| [Lumma](https://huggingface.co/collections/FrontiersMind/lumma-decision-models-6ab6356434b3db6856ec7ae6) | テキスト、画像 | 1.7× | 7.1× | 1.2× |
| [Lychee](https://huggingface.co/LycheeAI/Lychee-27B) | テキスト | 1.4× | 1.5× | 1.4× |
| [mammoth](https://huggingface.co/baselabs/mammoth-1) | テキスト、画像 | 3.3 to 5.3× | 2.3 to 3.4× | 1.6 to 2.0× |
| [mogan](https://huggingface.co/moganai/mogan-decision-31B-it) | テキスト | 1.6× | 8.5× | 1.8× |
| [OneJev](https://huggingface.co/collections/OmniJev/onejev16-6ad27669cb868abe0516486d) | テキスト、画像、動画 | 3.2 to 11.9× | 7.3× | 4.7× |
| [Open-Jev](https://huggingface.co/collections/ZefanCai/open-jev-6ab049b9d43a267bae4dedc8) | テキスト | 4.2× | 5.0× | 1.3 to 1.7× |
| [pplx-decider](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2) | テキスト、画像 | 3.1× | 1.4× | 1.2× |
| [Quyet](https://huggingface.co/collections/chinhnc/quyet-10-6ac1b597a9a6a6806bb01fa6) | テキスト | 1.9 to 5.9× | 7.0 to 7.2× | 1.4 to 1.9× |
| [Qwen3.5](https://huggingface.co/collections/Qwen/qwen35-6992e3053c019221cf2d725f) | テキスト、画像 | 1.2 to 3.3× | 2.4 to 7.7× | 1.0 to 2.0× |
| [Senova](https://huggingface.co/WangSenQWQ/Senova-4B) | テキスト | 3.1× | 28.1× | 1.2× |
| [Sieve](https://huggingface.co/collections/sthanika-ai/sieve-6abb93a61dd495f69811c586) | テキスト | 3.0× | 14.3× | 2.6× |
| [StartLux](https://huggingface.co/collections/startlux-models/startlux-decision-v11-instant-6aca8158623e8e001724cc82) | テキスト | 2.6× | 1.1× | 1.2× |
| [torchcast](https://huggingface.co/torchcast-ai/torchcast-decision-27b) | テキスト、画像 | 1.6× | 8.4× | 1.4× |

## オプション

| オプション | 内容 |
|---|---|
| `--session-cache-gb 40` | 以前のリクエストの状態を保存し、エージェントの次のステップは新しい部分だけを読む |
| `--batch-invariant` | どのリクエストと同じバッチでも、同じリクエストはビット単位で同じ結果になる（少し遅くなる） |
| `--quantize fp8` | Ada、Hopper、Blackwell の GPU で Qwen3.5 を FP8 で動かす |
| `--frontends 2` | HTTP、レンダリング、トークン化を 2 つの専用プロセスで行う（クライアントが多いとき向け） |

## ライセンス

Apache-2.0。`flide/kernels/gdn.py` には [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)（MIT）の 2 つのカーネルから派生したカーネルが含まれます。THIRD_PARTY_NOTICES に、それと Flide が利用しているほかのコードを記載しています。
