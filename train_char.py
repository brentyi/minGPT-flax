"""Script for training a GPT model on some text corpus. Pass in --help flag for options."""
import dataclasses
import pathlib

import dcargs
import fifteen
import jax
import jax_dataclasses as jdc
from tqdm.auto import tqdm

from mingpt import data, model, trainer


@dataclasses.dataclass
class TrainConfig:
    dataset_path: pathlib.Path
    experiment_name: str = "char_" + fifteen.utils.timestamp()
    restore_checkpoint: bool = False
    max_epochs: int = 1000
    minibatch_size: int = 128
    block_size: int = 128


def make_train_state(vocab_size: int, block_size: int) -> trainer.TrainState:
    """Initialize a training state for a given dataset.
    This includes the model, parameters, optimizer, etc.

    GPT and optimizer configurations copied from Karpathy's "play_char" notebook.
    """
    gpt_config = model.GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_head=8,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        n_layer=8,
        embd_dim=512,
        embd_pdrop=0.1,
    )
    optimizer_config = trainer.OptimizerConfig(
        learning_rate=6e-4,
        lr_decay=True,
        warmup_tokens=512 * 20,
        final_tokens=2 * 10 * block_size,
    )
    return trainer.TrainState.initialize(
        seed=0,
        gpt_config=gpt_config,
        optimizer_config=optimizer_config,
    )


def main(train_config: TrainConfig) -> None:
    experiment = fifteen.experiments.Experiment(
        data_dir=pathlib.Path("./experiments/") / train_config.experiment_name
    )
    devices = jax.local_devices()
    device_count = jax.local_device_count()
    assert (
        train_config.minibatch_size % device_count == 0
    ), f"Batch size {train_config.minibatch_size} must be divisible by {device_count}."

    # Load dataset.
    with open(train_config.dataset_path, "r") as f:
        train_dataset = data.CharDataset(
            data=f.read()[:-1],
            block_size=train_config.block_size,
        )

    # Write some metadata -- will need these to recreate the model at eval time.
    experiment.write_metadata("block_size", train_config.block_size)
    experiment.write_metadata("stoi", train_dataset.stoi)
    experiment.write_metadata("itos", train_dataset.itos)

    # Initialize training state, and replicate across each GPU.
    train_state = make_train_state(
        vocab_size=train_dataset.vocab_size,
        block_size=train_dataset.block_size,
    )
    if train_config.restore_checkpoint:
        train_state = experiment.restore_checkpoint(train_state)
    sharded_train_state: trainer.TrainState = jax.device_put_replicated(
        train_state, devices=devices
    )
    del train_state

    # Give each device a different PRNG key; this makes dropout masks unique.
    with jdc.copy_and_mutate(sharded_train_state) as sharded_train_state:
        sharded_train_state.prng_key = jax.random.split(
            sharded_train_state.prng_key[0], num=device_count
        )

    # Run training loop.
    train_dataloader = fifteen.data.DataLoader(
        dataset=train_dataset,
        minibatch_size=train_config.minibatch_size,
        num_workers=0,  # The entire dataset is in-memory, so we can skip parallelism.
    )
    for epoch in range(train_config.max_epochs):
        # Read training state from device 0 and save a checkpoint.
        train_state = jax.tree_map(lambda x: x[0], sharded_train_state)
        experiment.save_checkpoint(train_state, step=int(train_state.steps))
        del train_state

        # Grab iterable over minibatches, which have leaf shapes of (minibatch_size, ...).
        minibatches = train_dataloader.minibatches(shuffle_seed=epoch)

        # Convert leaf shapes to (device_count, minibatch_size // device_count, ...),
        # and prefetch to (potentially) improve parallelization.
        minibatches = fifteen.data.sharding_map(minibatches, devices=devices)
        minibatches = fifteen.data.prefetching_map(minibatches)

        for minibatch in tqdm(minibatches):
            # Training step.
            (
                sharded_train_state,
                sharded_log_data,
            ) = sharded_train_state.parallelized_train_step(minibatch)

            # Log to Tensorboard.
            experiment.log(
                sharded_log_data.fix_sharded_scalars(),
                step=sharded_train_state.steps[0],  # Pull step count from device 0.
                log_scalars_every_n=10,
                log_histograms_every_n=50,
            )

    # Read training state from device 0 and save a checkpoint.
    train_state = jax.tree_map(lambda x: x[0], sharded_train_state)
    experiment.save_checkpoint(train_state, step=int(train_state.steps))
    del train_state


if __name__ == "__main__":
    fifteen.utils.pdb_safety_net()
    main(dcargs.parse(TrainConfig))
