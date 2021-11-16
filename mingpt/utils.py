import random

import numpy as onp


def set_seed(seed: int) -> None:
    random.seed(seed)
    onp.random.seed(seed)
