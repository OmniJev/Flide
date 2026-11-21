"""Serving forwards one token budget to both the engine and online scheduler."""
import argparse
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.api import serve


class _Family:
    name = "fake"

    def __init__(self):
        pass


class ServeBudgetTests(unittest.TestCase):
    def parse(self, *options):
        parser = argparse.ArgumentParser()
        serve.add_parser(parser.add_subparsers())
        return parser.parse_args(["serve", "--model", "m", *options])

    def test_one_budget_reaches_both_engine_and_batcher(self):
        for explicit in (None, 16384, 65536, 204800):
            with self.subTest(explicit=explicit), contextlib.ExitStack() as stack:
                options = [] if explicit is None else ["--max-batch-tokens", str(explicit)]
                args = self.parse(*options)
                expected = 65536 if explicit is None else explicit
                stack.enter_context(patch("flide.runtime.engine.resolve_model", return_value="m"))
                stack.enter_context(patch("flide.runtime.engine.model_config"))
                stack.enter_context(patch("flide.formats.detect", return_value=_Family))
                engine = stack.enter_context(patch("flide.runtime.engine.Engine"))
                batcher = stack.enter_context(patch("flide.runtime.online.Batcher"))
                stack.enter_context(patch("flide.api.server.create_app"))
                stack.enter_context(patch("flide.api.serve.freeze_heap"))
                http_server = stack.enter_context(patch("uvicorn.run"))
                serve.run(args)
                self.assertEqual(engine.call_args.kwargs["budget"], expected)
                self.assertEqual(batcher.call_args.args[0], engine.return_value)
                self.assertEqual(batcher.call_args.kwargs["max_round_tokens"], expected)
                http_server.assert_called_once()

    def test_nonpositive_batch_budgets_are_rejected(self):
        for value in ("0", "-1"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.parse("--max-batch-tokens", value)
                self.assertEqual(error.exception.code, 2)

    def test_separate_serve_budget_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                self.parse("--budget", "102400")
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
