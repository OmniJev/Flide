"""The this-that-model layout without a checkpoint: the prompt text and the slot count.

    python tests/test_thisthat_format.py

The real token comparison against the release's own prompt.py needs the checkpoint's tokenizer; this pins
the layout itself, which is what the family has to reproduce.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugins"))

from flide.api.schema import SystemOneRequest

import flide_thisthat as family


class StubTokenizer:
    """One id per distinct string, so a decoded id is the exact text the layout asked for."""

    def __init__(self):
        self._ids, self._text = {}, {}

    def encode(self, text, add_special_tokens=False):
        if text not in self._ids:
            self._ids[text] = len(self._ids)
            self._text[self._ids[text]] = text
        return [self._ids[text]]

    def decode(self, ids):
        return "".join(self._text[i] for i in ids)


def main():
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}{'' if ok else f' ({detail})'}")

    tok = StubTokenizer()
    one = family.build(tok, "the state", [("Is it urgent?", ["no", "yes"])])
    check("one question ends with its answer slot", tok.decode(one["ids"]).endswith("\nAnswer: ("),
          repr(tok.decode(one["ids"])[-24:]))
    check("one question has one slot at the last token", one["slots"] == [len(one["ids"]) - 1], one["slots"])

    two = family.build(tok, "s", [("first?", ["a", "b"]), ("second?", ["c", "d"])])
    text = tok.decode(two["ids"])
    check("two questions are numbered", "Question 1: first?" in text and "Question 2: second?" in text, text)
    check("two questions get two slots, interleaved", len(two["slots"]) == 2 and two["slots"][0] < two["slots"][1],
          two["slots"])
    check("the second slot is the last token", two["slots"][1] == len(two["ids"]) - 1, two["slots"])

    wide = family.build(tok, "s", [("pick", [f"o{i}" for i in range(12)])])
    check("more than ten options still render", len(wide["ids"]) > 3 and wide["n_options"] == [12], wide["n_options"])

    body = {"state": "s", "questions": {
        "u": {"type": "noul", "instructions": "urgent?", "criteria": {"true": "it holds", "false": "it does not"}},
        "c": {"type": "choice", "instructions": "pick", "criteria": {"x": "first", "y": "second"}},
        "r": {"type": "score", "instructions": "rate", "criteria": ["low", "mid", "high"]}}}
    req = SystemOneRequest.model_validate(body)
    noul = family.typed("u", req.questions["u"])
    check("noul puts the criteria in the text and no/yes in the labels",
          noul[0].endswith("- no: it does not\n- yes: it holds") and noul[1] == ["no", "yes"], noul[:2])
    choice = family.typed("c", req.questions["c"])
    check("choice scores the criteria keys", choice[1] == ["x", "y"] and choice[2] == ["x", "y"], choice[1:3])
    score = family.typed("r", req.questions["r"])
    check("score scores 0..n-1", score[1] == ["0", "1", "2"] and score[2] == ["0", "1", "2"], score[1:3])

    # the readout: probabilities from the slot rows, and the score the release reports (its argmax level)
    plan = family.Plan([], [[1, 2, 3]], (family.Read(family.READ_HEAD, positions=(-1,)),), [
        ("u", "noul", ("no", "yes"), ("no", "yes")),
        ("r", "score", ("0", "1", "2"), ("0", "1", "2")),
    ])
    rows = [[0.0, 2.0], [0.0, 1.0, 4.0]]      # noul: yes wins; score: level 2 wins by a wide margin
    req = SystemOneRequest.model_validate({"state": "s", "questions": {
        "u": {"type": "noul", "instructions": "is it?"},
        "r": {"type": "score", "instructions": "rate", "criteria": ["low", "mid", "high"]}}})
    fam = family.ThisThat.__new__(family.ThisThat)
    fam.temperature = family.DEFAULT_TEMPERATURE
    fam.layout = "state_first"
    response, _ = fam.finish(req, plan, [[[rows[0]]], [[rows[1]]]])   # one branch per question
    expected_yes = 1.0 / (1.0 + math.exp(-2.0 / family.DEFAULT_TEMPERATURE))   # the logits are divided by tau
    check("noul reads P(yes) at the second label", abs(response.answers["u"].noul - expected_yes) < 1e-5,
          response.answers["u"].noul)
    check("score reports the argmax level like the release", response.answers["r"].score == 2.0,
          response.answers["r"].score)
    check("score keeps the whole distribution", len(response.answers["r"].probabilities) == 3,
          response.answers["r"].probabilities)

    # the schema_first layout: the questions are a state-independent prefix, the slots sit after the state
    first = family.build(tok, "state one", [("pick?", ["a", "b"])], layout="schema_first")
    second = family.build(tok, "state two", [("pick?", ["a", "b"])], layout="schema_first")
    check("schema_first starts with the question", tok.decode(first["ids"]).startswith("Question: pick?"),
          tok.decode(first["ids"])[:24])
    check("the question prefix does not depend on the state",
          first["ids"][: first["prefix_len"]] == second["ids"][: second["prefix_len"]], first["prefix_len"])
    check("the prefix ends before the context", tok.decode(first["ids"][first["prefix_len"]:]).startswith("\n\nContext:"))
    check("the slots still point at the answer cues",
          all(tok.decode([first["ids"][at]]).endswith("(") for at in first["slots"]), first["slots"])

    fam_schema = family.ThisThat.__new__(family.ThisThat)
    fam_schema.tokenizer = tok
    fam_schema.max_state_tokens = family.DEFAULT_MAX_STATE_TOKENS
    fam_schema.temperature, fam_schema.layout = family.DEFAULT_TEMPERATURE, "schema_first"
    req2 = SystemOneRequest.model_validate({"state": "s", "questions": {
        "u": {"type": "noul", "instructions": "is it?"},
        "c": {"type": "choice", "instructions": "pick", "criteria": {"x": "first", "y": "second"}}}})
    plan2 = fam_schema.plan(req2)
    check("schema_first keeps the questions as the plan prefix", len(plan2.prefix) > 0, len(plan2.prefix))
    check("schema_first reads every slot from one branch", len(plan2.suffixes) == 1
          and len(plan2.reads[0].positions) == 2, (len(plan2.suffixes), plan2.reads))
    response2, _ = fam_schema.finish(req2, plan2, [[rows]])        # one branch holding every slot
    check("schema_first maps one row per question", response2.answers["u"].noul == response.answers["u"].noul)

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()