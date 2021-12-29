"""Data helpers, adapted from Andrej Karpathy's minGPT repo:
https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""

from typing import Dict

import numpy as onp


class CharDataset:
    """Character dataset. Based on code from Karpathy's `play_char.ipynb`."""

    def __init__(self, data: str, block_size: int):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print("data has %d characters, %d unique." % (data_size, vocab_size))

        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}
        self.block_size: int = block_size
        self.vocab_size: int = vocab_size
        self.data: onp.ndarray = onp.array(
            [self.stoi[s] for s in data], dtype=onp.int32
        )

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> onp.ndarray:
        indices = self.data[idx : idx + self.block_size + 1]
        assert indices.shape == (self.block_size + 1,)
        return indices
