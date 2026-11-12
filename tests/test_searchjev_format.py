"""The SearchJev family without a checkpoint: prompt layout, option mapping, labels, temperatures.

    python tests/test_searchjev_format.py

The token comparison against the release's own PromptBuilder needs its tokenizer; this pins the pieces that do
not.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugins"))

from flide.api.schema import SystemOneRequest

import flide_searchjev as family


class StubTokenizer:
    """One id per distinct string, so a decoded id is the exact text the layout asked for."""

    def __init__(self, single=()):
        self._ids, self._text, self.chat_template = {}, {}, None
        self._single = set(single)

    def encode(self, text, add_special_tokens=False):
        if self._single and text not in self._single:
            return [0, 1]                       # only the listed labels are single tokens
        if text not in self._ids:
            self._ids[text] = len(self._ids)
            self._text[self._ids[text]] = text
        return [self._ids[text]]

    def decode(self, ids):
        return "".join(self._text.get(i, "") for i in ids)


def main():
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}{'' if ok else f' ({detail})'}")

    check("noul options are the release's two fixed texts", family.noul_options() ==
          [("true", "成立 / True"), ("false", "不成立 / False")], family.noul_options())
    check("noul criteria are appended to the option text",
          family.noul_options({"true": "needs action", "false": "can wait"})[0][1] == "成立 / True：needs action")
    check("a letter key shows only its description", family.option_text("A", "relevant") == "relevant")
    check("a word key keeps the key", family.option_text("rel", "relevant") == "rel：relevant")
    check("score levels read i: description",
          family.build_options("score", ["none", "low"]) == [("0", "0: none"), ("1", "1: low")])
    check("dict state renders one line per entry", family.render_state({"a": 1, "b": "x"}) == "a: 1\nb: x",
          family.render_state({"a": 1, "b": "x"}))
    check("a JSON string state is rendered through the same path",
          family.render_state('{"a": 1}') == "a: 1", family.render_state('{"a": 1}'))
    check("a chat list renders as turns",
          family.render_state([{"role": "user", "content": "hi"}]) == "[user] hi",
          family.render_state([{"role": "user", "content": "hi"}]))

    tok = StubTokenizer()
    fmt = family.SearchJev.__new__(family.SearchJev)
    fmt.tokenizer, fmt.language, fmt.style = tok, "bilingual", "plain"
    fmt.max_length = 2048
    fmt.system_prompt = family.SYSTEM_PROMPTS["bilingual"]
    fmt.hints = family.TYPE_HINTS["bilingual"]
    fmt.answer_line = family.ANSWER_INSTRUCTIONS["bilingual"]
    fmt.labels = list("ABCDEFGH")
    fmt.temperature = None
    ids = fmt._compose("the state", "noul", "Is it urgent?", family.noul_options())
    text = tok.decode(ids)
    check("the prompt opens the state inside the untrusted markers",
          "### State\n<<<STATE\nthe state\nSTATE>>>" in text, text[:80])
    check("the question carries its type hint", "[命题判断 / true or false] Is it urgent?" in text)
    check("the options are labelled from A", "A. 成立 / True\nB. 不成立 / False" in text)
    check("the plain style ends with the answer cue", text.endswith("Answer with exactly one option label.\n\nAnswer:\n"),
          repr(text[-48:]))

    body = {"state": "s", "questions": {
        "u": {"type": "noul", "instructions": "is it?"},
        "c": {"type": "choice", "instructions": "pick", "criteria": {"x": "first", "y": "second"}},
        "r": {"type": "score", "instructions": "rate", "criteria": ["low", "high"]}}}
    req = SystemOneRequest.model_validate(body)
    plan = fmt.plan(req)
    check("one branch per question", len(plan.suffixes) == 3, len(plan.suffixes))
    check("each branch reads its own last token", all(r.head == family.READ_HEAD for r in plan.reads), plan.reads)
    check("the data keeps the option keys", [d[2] for d in plan.data] ==
          [("true", "false"), ("x", "y"), ("0", "1")], [d[2] for d in plan.data])

    fmt.calib = {"noul": 0.5, "global": 1.5}
    check("a flat per-type temperature wins", fmt._temperature("noul") == 0.5, fmt._temperature("noul"))
    check("a missing type falls back to global", fmt._temperature("choice") == 1.5, fmt._temperature("choice"))
    fmt.calib = {}
    check("no calibration means temperature 1", fmt._temperature("noul") == 1.0)
    fmt.temperature = 2.0
    check("an explicit temperature wins over calibration", fmt._temperature("noul") == 2.0)

    kept, ids_ = family.label_space(StubTokenizer(single={"A", "B", "C"}), ["A", "B", "Ψ"], lenient=True)
    check("labels that are not one token are skipped", kept == ["A", "B"], kept)

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()