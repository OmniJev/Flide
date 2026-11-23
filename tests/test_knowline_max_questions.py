"""KnowLine question limits and default prompts, using an entirely local tokenizer."""
import json
import string
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugins"))

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from flide.api.schema import SystemOneRequest
from flide_knowline import KnowLine, MAX_QUESTIONS


def tokenizer():
    labels = list(string.ascii_uppercase)
    labels += [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    words = ["[UNK]", "user:", "assistant:", "Answer:", *labels]
    backend = Tokenizer(models.WordLevel(dict(zip(words, range(len(words)))), unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                         "{% if add_generation_prompt %}assistant: {% endif %}")
    return tok


def request(count):
    return SystemOneRequest.model_validate({"state": "A customer needs help.", "questions": {
        f"q{i}": {"type": "choice", "instructions": f"Question {i}",
                  "criteria": {"yes": "Help the customer.", "no": "Ignore the customer."}}
        for i in range(count)}})


class QuestionLimits(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        Path(self.directory.name, "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}))
        self.tokenizer = tokenizer()

    def family(self, **options):
        return KnowLine(self.directory.name, tokenizer=self.tokenizer, read_twice=False, **options)

    def test_default_remains_64(self):
        family = self.family()
        self.assertEqual(family.max_questions, MAX_QUESTIONS)
        self.assertEqual(MAX_QUESTIONS, 64)
        self.assertEqual(len(family.plan(request(64)).suffixes), 64)
        with self.assertRaisesRegex(ValueError, "65 questions, limit is 64"):
            family.plan(request(65))

    def test_configured_limit_reaches_planning_and_answers(self):
        family = self.family(max_questions="255")
        req = request(255)
        plan = family.plan(req)
        self.assertEqual(len(plan.suffixes), 255)
        response, meta = family.finish(req, plan, [[[[2.0, 0.0]]] for _ in range(255)])
        self.assertEqual(set(response.answers), set(req.questions))
        self.assertTrue(all(a.choice == "yes" for a in response.answers.values()))
        self.assertEqual(response.usage.output_tokens, 255)
        self.assertEqual(meta["branches"], 255)
        with self.assertRaisesRegex(ValueError, "256 questions, limit is 255"):
            family.plan(request(256))

    def test_lower_limit_and_empty_request(self):
        family = self.family(max_questions=1)
        self.assertEqual(len(family.plan(request(1)).suffixes), 1)
        with self.assertRaisesRegex(ValueError, "2 questions, limit is 1"):
            family.plan(request(2))
        with self.assertRaisesRegex(ValueError, "0 questions, limit is 1"):
            family.plan(SystemOneRequest.model_construct(state="", questions={}))

    def test_invalid_configuration(self):
        for limit in (0, -1):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "max_questions must be positive"):
                self.family(max_questions=limit)

    def test_explicit_default_preserves_plan_and_answers(self):
        default, explicit = self.family(), self.family(max_questions=64)
        req = request(3)
        left, right = default.plan(req), explicit.plan(req)
        self.assertEqual(left, right)
        outputs = [[[[2.0, 0.0]]] for _ in range(3)]
        self.assertEqual(default.finish(req, left, outputs), explicit.finish(req, right, outputs))


if __name__ == "__main__":
    unittest.main()
