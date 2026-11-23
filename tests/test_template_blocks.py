"""A template can fork a bidirectional state with separately tokenized branches."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.api.schema import SystemOneRequest
from flide.formats.base import Read
from flide.formats.template import Template


class StubTokenizer:
    all_special_tokens = ["<S>", "<END>", "<PAD>", "<END>"]

    def __init__(self):
        self.calls = []

    def __call__(self, texts, *, add_special_tokens):
        assert not add_special_tokens
        self.calls.append(texts)
        return {"input_ids": [list(map(ord, text)) for text in texts]}


def main():
    request = SystemOneRequest.model_validate({
        "state": {"document": "shared"},
        "questions": {
            "first": {"type": "noul", "instructions": "relevant"},
            "second": {"type": "noul", "instructions": "complete"},
        },
    })
    spec = {"labels": ["A", "B"], "label_ids": [1, 2], "types": ["noul"],
            "prompt": "{{ question.instructions }}{{ special_tokens[0] }}"}
    tokenizer = StubTokenizer()
    family = Template(".", tokenizer, spec={**spec, "attention": "blocks", "shared_prefix": "{{ state.document }}",
                                             "round_noul": False, "count_output_tokens": False})
    assert family.prompt.globals["special_tokens"] == ("<END>", "<PAD>", "<S>")
    plan = family.plan(request)
    assert tokenizer.calls == [["shared", "relevant<END>", "complete<END>"]]
    assert plan.attention == "blocks" and plan.reads == (Read("labels"),)
    assert plan.prefix == list(map(ord, "shared"))
    assert plan.suffixes == [list(map(ord, text)) for text in ["relevant<END>", "complete<END>"]]
    result, _ = family.finish(request, plan, [[[[0.0, 2.0]]], [[[0.0, -2.0]]]])
    assert math.isclose(result.answers["first"].noul, 1.0 / (1.0 + math.exp(-2.0)))
    assert math.isclose(result.answers["second"].noul, 1.0 / (1.0 + math.exp(2.0)))
    assert result.usage.output_tokens == 0

    separate = Template(".", tokenizer, spec={**spec, "attention": "bidirectional"})
    plan = separate.plan(request)
    assert plan.attention == "blocks" and plan.prefix == []
    result, _ = separate.finish(request, plan, [[[[0.0, 2.0]]], [[[0.0, -2.0]]]])
    assert result.usage.output_tokens == 2 and result.answers["first"].noul == 0.880797
    print("PASS template shared blocks, separate bidirectional prompts, and binary readout")


if __name__ == "__main__":
    main()
