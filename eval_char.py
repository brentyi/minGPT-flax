"""Script for autoregressive sampling from our model. Pass in --help flag for options."""

import dataclasses
from typing import Any, Dict, Tuple, Union, cast

import dcargs
import fannypack
import jax
import numpy as onp
from jax import numpy as jnp
from tqdm.auto import tqdm

from mingpt import experiment_files, trainer
from train_char import make_train_state

PRNGKey = Union[Any, jnp.ndarray]


@dataclasses.dataclass
class Args:
    experiment_name: str
    sample_steps: int = 500
    sample_from_top_k: int = 3


def sample(
    train_state: trainer.TrainState,
    initial_conditioner: onp.ndarray,
    steps: int,
    prng_key: PRNGKey,
    temperature: float = 1.0,
    sample_from_top_k: int = 1,
) -> onp.ndarray:
    """Sample from GPT model. Conditioner input and output are both integer indices."""

    block_size: int = train_state.model.config.block_size
    vocab_size: int = train_state.model.config.vocab_size

    assert (
        len(initial_conditioner.shape) == 1 and initial_conditioner.dtype == onp.int32
    )
    conditioner_length = initial_conditioner.shape[0]

    out = onp.zeros(conditioner_length + steps, dtype=onp.int32)
    out[:conditioner_length] = initial_conditioner

    for i in tqdm(range(conditioner_length, conditioner_length + steps)):
        padded_conditioner = onp.zeros((1, block_size), dtype=onp.int32)

        if i <= block_size:
            padded_conditioner[0, :i] = out[:i]
            predicted_token_idx = i - 1
        else:
            padded_conditioner[0, :] = out[i - block_size : i]
            predicted_token_idx = -1

        logits = train_state.predict(cast(jnp.ndarray, padded_conditioner))

        assert logits.shape == (1, block_size, vocab_size)
        logits_onp = onp.array(logits[0, predicted_token_idx, :]) / temperature
        not_top_k_indices = onp.argpartition(logits_onp, -sample_from_top_k)[
            :-sample_from_top_k
        ]
        logits_onp[not_top_k_indices] = -onp.inf

        out[i], prng_key = sample_from_logits(
            cast(jnp.ndarray, logits_onp), prng_key=prng_key
        )

    return out


@jax.jit
def sample_from_logits(
    logits: jnp.ndarray, prng_key: PRNGKey
) -> Tuple[jnp.ndarray, PRNGKey]:
    """Helper for categorical sampling + propagating PRNG keys."""

    assert len(logits.shape) == 1
    key0, key1 = jax.random.split(prng_key)
    return jax.random.categorical(key0, logits), key1


def main(args: Args):

    fannypack.utils.pdb_safety_net()

    experiment = experiment_files.ExperimentFiles(
        identifier=args.experiment_name
    ).assert_exists()

    # Read model metadata.
    block_size = experiment.read_metadata("block_size", int)
    stoi: Dict[str, int] = experiment.read_metadata("stoi", dict)
    itos: Dict[int, str] = experiment.read_metadata("itos", dict)

    def index_array_from_string(string: str) -> onp.ndarray:
        return onp.array([stoi[char] for char in string], dtype=onp.int32)

    def string_from_index_array(array: onp.ndarray) -> str:
        assert len(array.shape) == 1
        return "".join([itos[idx] for idx in array])

    # Restore training state.
    train_state = make_train_state(vocab_size=len(stoi), block_size=block_size)
    train_state = experiment.restore_checkpoint(train_state)
    print("Loaded checkpoint at step:", train_state.steps)

    # Generate samples from prompts.
    while True:
        print("Enter conditioner:")
        conditioner = input()

        out = sample(
            train_state,
            initial_conditioner=index_array_from_string(conditioner),
            steps=args.sample_steps,
            prng_key=jax.random.PRNGKey(0),
            sample_from_top_k=args.sample_from_top_k,
        )
        print(string_from_index_array(out))
        print()


if __name__ == "__main__":
    main(dcargs.parse(Args))
