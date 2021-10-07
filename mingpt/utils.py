"""Utilities, mostly copied and pasted from:
https://github.com/brentyi/dfgo/blob/master/lib/utils.py
"""
import argparse
import dataclasses
import random
from typing import Iterable, Optional, Type, TypeVar

import jax
import numpy as onp

PytreeType = TypeVar("PytreeType")
DataclassType = TypeVar("DataclassType")


def set_seed(seed):
    random.seed(seed)
    onp.random.seed(seed)


def collate_fn(batch: Iterable[PytreeType], axis: int = 0) -> PytreeType:
    """Collate function for torch DataLoaders."""
    return jax.tree_map(lambda *arrays: onp.stack(arrays, axis=axis), *batch)
