"""`flide serve --budget` reaches the engine; the online rounds keep their own size (one flat replay with the CUDA
graphs, the engine's budget without them)."""
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

    def test_budget_reaches_the_engine(self):
        for explicit in (None, 16384, 65536, 204800):
            with self.subTest(explicit=explicit), contextlib.ExitStack() as stack:
                options = [] if explicit is None else ["--budget", str(explicit)]
                args = self.parse(*options)
                expected = explicit   # None: the engine picks it by the card's memory
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
                self.assertNotIn("max_round_tokens", batcher.call_args.kwargs)
                http_server.assert_called_once()

    def test_nonpositive_budgets_are_rejected(self):
        for value in ("0", "-1"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.parse("--budget", value)
                self.assertEqual(error.exception.code, 2)

    def test_no_second_budget_option(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                self.parse("--max-batch-tokens", "102400")
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
