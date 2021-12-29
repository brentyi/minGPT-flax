"""GPT implementation in Flax, based on Andrej Karpathy's minGPT (PyTorch):
https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""

import dataclasses
from functools import partial

import numpy as onp
from flax import linen as nn
from jax import numpy as jnp


@dataclasses.dataclass
class GPTConfig:
    # Input size config.
    vocab_size: int
    block_size: int

    # Transformer block config.
    n_head: int
    resid_pdrop: float
    attn_pdrop: float

    # Overall config.
    n_layer: int
    embd_dim: int
    embd_pdrop: float

    @staticmethod
    def make_gpt1_config(vocab_size: int, block_size: int) -> "GPTConfig":
        """Helper for defining GPT-1 like networks, roughly 125M params."""

        return GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            # Transformer block config.
            n_head=12,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
            # Overall config>
            n_layer=12,
            embd_dim=768,
            embd_pdrop=0.1,
        )


# GPT quote: "Since layernorm is used extensively throughout the model, a simple weight
# initialization of N(0,0.02) was sufficient."
# TODO: why 0.02? why not 0.01? or 0.03? or 1.0?
DenseWithInit = partial(
    nn.Dense,
    kernel_init=nn.initializers.normal(stddev=0.02),
    bias_init=nn.initializers.zeros,
)
EmbedWithInit = partial(
    nn.Embed,
    embedding_init=nn.initializers.normal(stddev=0.02),
)


class CausalSelfAttention(nn.Module):
    """A simple masked self-attention module. In practice it probably makes more sense
    to use `flax.linen.attention.*`.
    """

    n_head: int
    resid_pdrop: float
    attn_pdrop: float
    embd_dim: int  # Not actually needed, just used for assertion on input shape.

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:  # type: ignore
        # Input shape should be (batch size, token count, channels).
        B, T, C = x.shape
        assert C == self.embd_dim

        # Token dimension should be the same as embedding dimension.
        assert (
            C % self.n_head == 0
        ), "Embedding size should be evenly divisible by head count"
        head_size: int = C // self.n_head

        # Apply linear mappings to compute key, query, values for all heads in the
        # batch.
        def kqv_linear_map(x: jnp.ndarray) -> jnp.ndarray:
            return DenseWithInit(features=C)(x).reshape((B, T, self.n_head, head_size))

        k = kqv_linear_map(x)
        q = kqv_linear_map(x)
        v = kqv_linear_map(x)
        assert k.shape == q.shape == v.shape == (B, T, self.n_head, head_size)

        # For computing QK^T:
        #     Inputs are: (B, T, nh, hs), (B, T, nh, hs)
        #     Desired output is: (B, nh, T, T)
        # Dividing by square root of the head size results in better gradients for
        # softmax (more values on the locally linear area).
        att = jnp.einsum("bihs,bkhs->bhik", k, q) / onp.sqrt(head_size)
        assert att.shape == (B, self.n_head, T, T)

        # Create and apply a causal mask.
        mask = onp.tril(onp.ones((T, T), dtype=bool)).reshape((1, 1, T, T))
        att = jnp.where(mask, att, -jnp.inf)
        assert att.shape == (B, self.n_head, T, T)

        # Softmax over last axis.
        att = nn.softmax(att, axis=-1)
        assert att.shape == (B, self.n_head, T, T)

        # Dropout.
        att = nn.Dropout(rate=self.attn_pdrop, deterministic=deterministic)(att)

        # For computing softmax(QK^T/sqrt(head_size)).
        #     Inputs are: (B, nh, T, T), (B, T, nh, hs)
        #     Desired output is: (B, T, nh, hs)
        # note that softmax was applied to the final T dimension of `att`!
        y = jnp.einsum("bnti,binh->btnh", att, v)
        y = y.reshape((B, T, C))

        # Output projection, dropout.
        y = DenseWithInit(features=C)(y)
        y = nn.Dropout(rate=self.resid_pdrop, deterministic=deterministic)(y)
        assert y.shape == x.shape == (B, T, C)

        return y


class GPTBlock(nn.Module):
    """An unassuming GPT block."""

    config: GPTConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:  # type: ignore
        x = x + CausalSelfAttention(
            n_head=self.config.n_head,
            resid_pdrop=self.config.resid_pdrop,
            attn_pdrop=self.config.attn_pdrop,
            embd_dim=self.config.embd_dim,
        )(nn.LayerNorm()(x), deterministic=deterministic)

        def mlp(x: jnp.ndarray) -> jnp.ndarray:
            x = DenseWithInit(features=4 * self.config.embd_dim)(x)
            x = nn.gelu(x, approximate=True)
            x = DenseWithInit(features=self.config.embd_dim)(x)
            x = nn.Dropout(rate=self.config.resid_pdrop, deterministic=deterministic)(x)
            return x

        x = x + mlp(nn.LayerNorm()(x))  # type: ignore
        return x


class GPT(nn.Module):
    """The full GPT language model, with a context size of `block_size`."""

    config: GPTConfig

    @nn.compact
    def __call__(self, idx: jnp.ndarray, deterministic: bool) -> jnp.ndarray:  # type: ignore
        B, T = idx.shape
        assert (
            T <= self.config.block_size
        ), "Cannot forward, model block size is exhausted"
        assert idx.dtype == jnp.int32

        # Compute token embeddings.
        token_embeddings = EmbedWithInit(
            num_embeddings=self.config.vocab_size,
            features=self.config.embd_dim,
        )(idx)
        assert token_embeddings.shape == (B, T, self.config.embd_dim)

        # Compute positional embeddings.
        pos_embedding_variable = self.variable(
            "params",
            "position_embeddings",
            jnp.zeros,  # TODO: get intuition for why this is initialized to zero? why not random normal?
            (self.config.block_size, self.config.embd_dim),  # shape arg for jnp.zeros
        )
        pos_embeddings = pos_embedding_variable.value[:T, :]
        assert pos_embeddings.shape == token_embeddings.shape[1:]

        # Apply transformer blocks.
        x = token_embeddings + pos_embeddings[None, :, :]
        x = nn.Dropout(rate=self.config.embd_pdrop, deterministic=deterministic)(x)
        for _ in range(self.config.n_layer):
            x = GPTBlock(config=self.config)(x, deterministic=deterministic)
        x = nn.LayerNorm()(x)
        logits = DenseWithInit(features=self.config.vocab_size, use_bias=False)(x)

        # Andrej's implementation also (optionally) computes a loss here, but this seems
        # like a weird place to do that.
        assert logits.shape == (B, T, self.config.vocab_size)
        return logits
