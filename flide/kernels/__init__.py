"""Triton kernels. Importing this package routes Triton's run-time C builds through prebuilt modules
(launcher.py), before any kernel here or in fla is compiled."""
from .launcher import install as _install

_install()
