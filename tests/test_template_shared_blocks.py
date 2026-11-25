"""Template attention layouts and their shared-prefix boundaries without a model or GPU."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.api.schema import SystemOneRequest
from flide.formats.template import Template
from flide.runtime.pack import pack
from flide.runtime.schedule import split_plan


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, str):
            return {"input_ids": list(map(ord, text))}
        return {"input_ids": [list(map(ord, row)) for row in text]}


MARKER = "<end>"
PROMPT = "{{ state }}<end>{{ boundary() }}{{ question.instructions }}"


def template(**options):
    spec = {"prompt": PROMPT, "labels": ["A", "B"], "label_ids": [65, 66], **options}
    return Template(".", tokenizer=CharacterTokenizer(), spec=spec)


def request(*questions):
    return SystemOneRequest.model_validate({
        "state": "material",
        "questions": {str(i): {"type": "noul", "instructions": text} for i, text in enumerate(questions)},
    })


class SharedBlocksTest(unittest.TestCase):
    def test_blocks_pack_one_prefix_for_two_questions(self):
        family = template(attention="blocks", prefix_ends=MARKER)
        plan = family.plan(request("first", "second"))
        alone = family.plan(request("first"))
        self.assertEqual(plan.attention, "blocks")
        self.assertEqual(plan.prefix, list(map(ord, "material" + MARKER)))
        self.assertEqual(plan.suffixes, [list(map(ord, text)) for text in ("first", "second")])
        self.assertEqual(alone.prefix, plan.prefix)
        self.assertEqual(alone.suffixes, plan.suffixes[:1])

        pieces = split_plan(0, plan, budget=plan.tokens)
        self.assertEqual(len(pieces), 1)
        self.assertTrue(pieces[0].job.blocks)
        batch = pack([pieces[0].job], "cpu")
        self.assertEqual(batch.n_prefix_tokens, len(plan.prefix))
        self.assertEqual(batch.n_tokens, len(plan.prefix) + len("firstsecond"))

    def test_existing_attention_modes(self):
        req = request("first", "second")
        causal = template().plan(req)
        prefix = template(attention="prefix", prefix_ends=MARKER).plan(req)
        independent = template(attention="bidirectional").plan(req)
        self.assertEqual(causal.attention, "causal")
        self.assertEqual(prefix.attention, "prefix")
        self.assertEqual(prefix.prefix, list(map(ord, "material" + MARKER)))
        self.assertEqual(independent.attention, "blocks")
        self.assertEqual(independent.prefix, [])
        self.assertEqual(independent.suffixes, [prefix.prefix + tail for tail in prefix.suffixes])

    def test_blocks_require_a_nonempty_text_prefix_marker(self):
        with self.assertRaisesRegex(ValueError, "prefix_ends"):
            template(attention="blocks")
        with self.assertRaisesRegex(ValueError, "nonempty prefix_ends"):
            template(attention="blocks", prefix_ends="")
        with self.assertRaisesRegex(ValueError, "no images"):
            Template(".", tokenizer=CharacterTokenizer(), spec={
                "attention": "blocks", "prefix_ends": MARKER, "messages": [], "images": {},
            })

    def test_blocks_reject_invalid_prompt_boundaries(self):
        family = template(attention="blocks", prefix_ends=MARKER)
        with self.assertRaisesRegex(ValueError, "different positions"):
            family.plan(request("first", "second<end> extra"))
        with self.assertRaisesRegex(ValueError, "prompts differ"):
            family.plan(request("first", ""))
        mismatched = template(attention="blocks", prefix_ends=MARKER,
                              prompt="{{ question.key }}" + PROMPT)
        with self.assertRaisesRegex(ValueError, "prompts differ"):
            mismatched.plan(request("first", "second"))
        missing = template(attention="blocks", prefix_ends="<missing>")
        with self.assertRaisesRegex(ValueError, "no end"):
            missing.plan(request("first"))


if __name__ == "__main__":
    unittest.main()
