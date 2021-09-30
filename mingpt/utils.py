"""Utilities, mostly copied and pasted from:
https://github.com/brentyi/dfgo/blob/master/lib/utils.py
"""
import argparse
import dataclasses
import random
from typing import Iterable, Optional, Type, TypeVar

import datargs
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


def parse_args(
    cls: Type[DataclassType], *, description: Optional[str] = None
) -> DataclassType:
    """Populates a dataclass via CLI args. Basically the same as `datargs.parse()`, but
    adds default values to helptext."""
    assert dataclasses.is_dataclass(cls)

    # Modify helptext to add default values.
    #
    # This is a little bit prettier than using the argparse helptext formatter, which
    # will include dataclass.MISSING values.
    for field in dataclasses.fields(cls):
        if field.default is not dataclasses.MISSING:
            # Heuristic for if field has already been mutated. By default metadata will
            # resolve to a mappingproxy object.
            if isinstance(field.metadata, dict):
                continue

            # Add default value to helptext!
            if hasattr(field.default, "name"):
                # Special case for enums
                default_fmt = f"(default: {field.default.name})"
            else:
                default_fmt = "(default: %(default)s)"

            field.metadata = dict(field.metadata)
            field.metadata["help"] = (
                f"{field.metadata['help']} {default_fmt}"
                if "help" in field.metadata
                else default_fmt
            )

    return datargs.parse(cls, parser=argparse.ArgumentParser(description=description))
