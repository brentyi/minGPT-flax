"""Training helpers for our GPT model.

Loosely based on code from Andrej Karpathy and Mikhail Grankin:
    https://github.com/karpathy/minGPT/blob/master/mingpt/trainer.py
    https://github.com/mgrankin/minGPT/blob/main/mingpt/trainer.py
"""

import dataclasses
import functools
from typing import Dict, Tuple, Union

import fifteen
import flax
import jax
import jax.flatten_util
import jax_dataclasses as jdc
import numpy as onp
import optax
from jax import numpy as jnp
from typing_extensions import Annotated

from .model import GPT, GPTConfig


@dataclasses.dataclass
class OptimizerConfig:
    learning_rate: float = 6e-4
    lr_decay: bool = True  # If decay is enabled, we use cosine annealing.

    adam_b1: float = 0.9
    adam_b2: float = 0.95

    # Token counts come from the GPT-3 paper, but may not be good defaults elsewhere.
    warmup_tokens: int = int(375e6)  # Tokens before reaching full learning rate.
    final_tokens: int = int(260e9)  # At what point we reach 10% of original LR

    weight_decay: float = 0.1  # L2 regularization coefficient.
    grad_norm_clip: float = 1.0

    @staticmethod
    def weight_decay_mask(params: flax.core.FrozenDict) -> flax.core.FrozenDict:
        def check_decay(
            subtree: flax.core.FrozenDict,
            parent_decay: bool,
        ) -> flax.core.FrozenDict:
            out: Dict[str, Union[bool, flax.core.FrozenDict]] = {}
            for k, v in subtree.items():
                assert isinstance(k, str)
                decay_based_on_key: bool = not any(
                    [exclude in k.lower() for exclude in ("embed", "layernorm", "bias")]
                )
                if isinstance(v, flax.core.FrozenDict):
                    out[k] = check_decay(
                        v,
                        parent_decay=(parent_decay and decay_based_on_key),
                    )
                else:
                    out[k] = parent_decay

            return flax.core.FrozenDict(out)

        out = check_decay(params, parent_decay=True)
        assert jax.tree_structure(out) == jax.tree_structure(params)
        assert isinstance(out, flax.core.FrozenDict)
        return out

    def make_optimizer_no_lr(self) -> optax.GradientTransformation:
        return optax.chain(
            optax.clip_by_global_norm(self.grad_norm_clip),
            optax.scale_by_adam(self.adam_b1, self.adam_b2),
            optax.masked(
                optax.add_decayed_weights(self.weight_decay),
                OptimizerConfig.weight_decay_mask,
            ),
        )

    def lr_scheduler(self, n_tokens: int) -> float:
        """Learning rate scheduler, adapted from Mikhail Grankin.

        This is excluded from the optax chain because we want to base our learning rate
        based on the token count, rather than the step count.

        https://github.com/mgrankin/minGPT/blob/main/mingpt/trainer.py
        """
        # Decay the learning rate based on our progress.
        if self.lr_decay:
            progress = (n_tokens - self.warmup_tokens) / max(
                1,
                # TODO: this will break if the difference exceeds the int32 bounds.
                # Seems unlikely, but actually happens with the default values above...
                self.final_tokens - self.warmup_tokens,
            )
            lr_mult = jnp.where(
                n_tokens < self.warmup_tokens,
                # Linear warmup.
                n_tokens / jnp.fmax(1, self.warmup_tokens),
                # Cosine learning rate decay.
                jnp.fmax(0.1, 0.5 * (1.0 + jnp.cos(onp.pi * progress))),
            )
            return self.learning_rate * lr_mult
        else:
            return self.learning_rate


@jdc.pytree_dataclass
class TrainState(jdc.EnforcedAnnotationsMixin):
    """GPT training state. Makes a somewhat strong assumption for learning rate
    scheduling: that we always use the same batch size and input token count."""

    model: GPT = jdc.static_field()
    params: flax.core.FrozenDict

    optimizer_unit_lr: optax.GradientTransformation = jdc.static_field()
    optimizer_state: optax.OptState
    optimizer_config: OptimizerConfig = jdc.static_field()

    prng_key: jax.random.KeyArray
    steps: Annotated[jnp.ndarray, (), jnp.integer]  # Scalar integer.

    @staticmethod
    def initialize(
        seed: int,
        gpt_config: GPTConfig,
        optimizer_config: OptimizerConfig,
    ) -> "TrainState":

        prng_key0, prng_key1 = jax.random.split(jax.random.PRNGKey(seed))

        # Initialize model.
        model = GPT(config=gpt_config)
        dummy_input = onp.zeros(
            # Shape is (B, T)... neither should matter.
            (1, 1),
            # Inputs are integer indices.
            dtype=jnp.int32,
        )
        params = model.init(prng_key0, dummy_input, deterministic=True)

        # Initialize optimizer.
        optimizer_unit_lr = optimizer_config.make_optimizer_no_lr()
        optimizer_state = optimizer_unit_lr.init(params)

        return TrainState(
            model=model,
            params=params,
            optimizer_unit_lr=optimizer_unit_lr,
            optimizer_state=optimizer_state,
            optimizer_config=optimizer_config,
            prng_key=prng_key1,
            steps=jnp.array(0),
        )

    @functools.partial(
        jax.pmap,
        axis_name="device",
        # Old training states are almost never reused, so we "donate" the argument to
        # let XLA reuse the memory, allowing parameters to be updated in-place. Note
        # that this will lead to runtime errors if an outdated training state object is
        # operated on.
        donate_argnums=(0,),
    )
    def sharded_train_step(
        self, minibatch: jnp.ndarray
    ) -> Tuple["TrainState", fifteen.experiments.TensorboardLogData]:
        """Perform a parallelized training step. The training state that this is called
        on should be sharded, eg already distributed on multiple devices.

        Returns the updated (sharded) training state."""
        return self._train_step(minibatch, is_pmapped=True)

    @functools.partial(
        jax.jit,
        # Old training states are almost never reused, so we "donate" the argument to
        # let XLA reuse the memory, allowing parameters to be updated in-place. Note
        # that this will lead to runtime errors if an outdated training state object is
        # operated on.
        donate_argnums=(0,),
    )
    def train_step(
        self, minibatch: jnp.ndarray
    ) -> Tuple["TrainState", fifteen.experiments.TensorboardLogData]:
        """Perform a training step on a single device.

        Returns the updated training state."""
        return self._train_step(minibatch, is_pmapped=False)

    def _train_step(
        self,
        minibatch: jnp.ndarray,
        is_pmapped: bool,
    ) -> Tuple["TrainState", fifteen.experiments.TensorboardLogData]:
        """Helper for generalizing training steps. `is_pmapped` can be used to
        enable/disable `jax.lax.pmean()` calls used for parameter synchronization."""

        # Batch size, token count.
        B = minibatch.shape[0]
        T = self.model.config.block_size
        assert minibatch.shape == (B, T + 1)
        assert jnp.issubdtype(
            minibatch.dtype, jnp.integer
        ), "Training inputs should be integer token indices!"

        # Offset inputs/outputs; goal is next-token prediction.
        x = minibatch[:, :-1]
        y = minibatch[:, 1:]

        # Split PRNG key.
        prng_key_dropout, prng_key_new = jax.random.split(self.prng_key)

        # Define cross-entropy loss.
        def compute_loss(
            params: flax.core.FrozenDict,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            y_pred_logits: jnp.ndarray = self.model.apply(
                params,
                x,
                deterministic=False,
                rngs={"dropout": prng_key_dropout},
            )
            y_label_one_hot = jax.nn.one_hot(
                y, num_classes=self.model.config.vocab_size
            )
            assert y_pred_logits.shape == y_label_one_hot.shape

            ce_loss = optax.softmax_cross_entropy(
                logits=y_pred_logits, labels=y_label_one_hot
            )
            assert ce_loss.shape == (B, T)
            return jnp.mean(ce_loss), y_pred_logits

        # Backprop + parameter update.
        (loss, y_pred_logits), grads = jax.value_and_grad(compute_loss, has_aux=True)(
            self.params
        )
        if is_pmapped:
            # Synchronize gradients across devices.
            grads = jax.lax.pmean(grads, axis_name="device")
        updates, optimizer_state_new = self.optimizer_unit_lr.update(
            grads, self.optimizer_state, self.params
        )

        # Apply learning rate scheduler. Note the negative sign needed to *minimize* the
        # loss.
        #
        # Somewhat strong assumption: we always use the same batch and token counts. We
        # could also explicitly track the token count, but when implemented naively
        # (with a uint32) this overflows pretty quickly.
        learning_rate = self.optimizer_config.lr_scheduler(
            n_tokens=jnp.array(self.steps, dtype=jnp.float32) * B * T
        )
        updates = jax.tree_map(lambda x: -learning_rate * x, updates)

        # Log data for Tensorboard.
        log_data = fifteen.experiments.TensorboardLogData(
            scalars={
                "train/loss": loss,
                "train/gradient_norm": optax.global_norm(grads),
                "train/learning_rate": learning_rate,
            },
            histograms={
                "train/y_pred_logits": y_pred_logits,
            },
        )

        if is_pmapped:
            # Synchronize scalars across devices.
            log_data = fifteen.experiments.TensorboardLogData(
                scalars=jax.lax.pmean(log_data.scalars, axis_name="device"),
                histograms=log_data.histograms,
            )

        # Return updated state.
        with jdc.copy_and_mutate(self) as state_new:
            state_new.params = optax.apply_updates(self.params, updates)
            state_new.optimizer_state = optimizer_state_new
            state_new.prng_key = prng_key_new
            state_new.steps += 1
        return state_new, log_data
