"""Script for training a GPT model on some text corpus. Pass in --help flag for options."""

import dataclasses
import dcargs
import datetime
import pathlib

import fannypack
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mingpt import data, experiment_files, model, trainer, utils


@dataclasses.dataclass
class TrainConfig:
    dataset_path: pathlib.Path
    experiment_name: str = "char_" + datetime.datetime.now().strftime(
        "%Y-%m-%d_%H:%M:%S"
    )
    max_epochs: int = 1000
    batch_size: int = 128
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
    fannypack.utils.pdb_safety_net()

    experiment = experiment_files.ExperimentFiles(
        identifier=train_config.experiment_name
    ).assert_new()

    # Block size = spatial extent of the model.
    with open(train_config.dataset_path, "r") as f:
        train_dataset = data.CharDataset(
            data=f.read()[:-1], block_size=train_config.block_size
        )

    # Write some metadata -- will need these to recreate the model at eval time.
    experiment.write_metadata("block_size", train_config.block_size)
    experiment.write_metadata("stoi", train_dataset.stoi)
    experiment.write_metadata("itos", train_dataset.itos)

    # Train!
    train_state = make_train_state(
        vocab_size=train_dataset.vocab_size,
        block_size=train_dataset.block_size,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_config.batch_size,
        num_workers=1,  # The entire dataset is in-memory
        collate_fn=utils.collate_fn,
        drop_last=True,
    )

    for epoch in range(train_config.max_epochs):

        # Save checkpoint at the start of each epoch. We could just do the learnable
        # parameters, but bundling up the whole training state is a bit easier.
        experiment.save_checkpoint(train_state, step=train_state.steps)

        for batch in tqdm(train_dataloader):
            x, y = batch
            train_state, log_data = train_state.training_step(x=x, y=y)

            # Log to Tensorboard.
            experiment.log(
                log_data,
                step=train_state.steps,
                log_scalars_every_n=10,
                log_histograms_every_n=50,
            )

    experiment.save_checkpoint(train_state, step=train_state.steps)


if __name__ == "__main__":
    main(dcargs.parse(TrainConfig))
