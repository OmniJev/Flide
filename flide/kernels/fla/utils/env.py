"""Copied from flash-linear-attention 0.6.0 (fla/utils/env.py), forward path only. Copyright (c) 2023-2026
Songlin Yang, Yu Zhang, Zhiyuan Li; MIT license (THIRD_PARTY_NOTICES)."""

import functools
import logging
import os
from functools import cache
from importlib.util import find_spec
import torch
import triton
from packaging import version as package_version


logger = logging.getLogger(__name__)


FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"


FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'


FLA_DISABLE_TENSOR_CACHE = os.getenv('FLA_DISABLE_TENSOR_CACHE', '0') == '1'


try:
    FLA_TENSOR_CACHE_SIZE = int(os.getenv('FLA_TENSOR_CACHE_SIZE', "4"))
except ValueError:
    FLA_TENSOR_CACHE_SIZE = 4


@cache
def check_pytorch_version(version_s: str = '2.4') -> bool:
    return package_version.parse(torch.__version__) >= package_version.parse(version_s)


TRITON_ABOVE_3_4_0 = package_version.parse(triton.__version__) >= package_version.parse("3.4.0")


TRITON_ABOVE_3_5_1 = package_version.parse(triton.__version__) >= package_version.parse("3.5.1")


TRITON_ABOVE_3_6_0 = package_version.parse(triton.__version__) >= package_version.parse("3.6.0")


TRITON_ABOVE_3_7_1 = package_version.parse(triton.__version__) >= package_version.parse("3.7.1")


TRITON_ABOVE_3_8_0 = package_version.parse(triton.__version__) >= package_version.parse("3.8.0")


@functools.cache
def find_spec_cached(name):
    try:
        return find_spec(name)
    except ModuleNotFoundError as error:
        if error.name is not None and (name == error.name or name.startswith(f'{error.name}.')):
            return None
        raise
