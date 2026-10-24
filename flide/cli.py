"""Command line: flide run | serve | kernels | formats."""
from __future__ import annotations

import argparse
import logging


def _formats(args) -> None:
    from . import formats

    for name in formats.names():
        cls = formats.get(name)
        print(f"{name}\t{cls.__module__}\tbackbones: {', '.join(cls.backbones) or 'any'}")


def main(argv=None) -> None:
    from .api import serve
    from .runtime import kcache, offline

    p = argparse.ArgumentParser(prog="flide")
    sub = p.add_subparsers(dest="command", required=True)
    offline.add_parser(sub)
    serve.add_parser(sub)
    kcache.add_parser(sub)
    f = sub.add_parser("formats", help="list installed model families")
    f.set_defaults(func=_formats)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
