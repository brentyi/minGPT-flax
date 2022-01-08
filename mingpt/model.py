"""GPT implementation in Flax, based on Andrej Karpathy's minGPT (PyTorch):
https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""
import dataclasses
from functools import partial
from typing import Optional

import jax
from flax import linen as nn
from jax import numpy as jnp

from . import attention


@dataclasses.dataclass
class GPTConfig:
    vocab_size: int

    # The history/context length of our sequence model.
    block_size: int

    n_head: int  # Output size for multi-headed self-attention.
    resid_pdrop: float  # Dropout probability.
    attn_pdrop: float  # Dropout probability.

    # Enable attention chunking to trade runtime for memory efficiency. We implement an
    # approach similar to the algorithm presented here:
    # https://arxiv.org/pdf/2112.05682v2.pdf
    #
    # If chunking is enabled, both q_chunk_size and kv_chunk_size must be set.
    # Note that `block_size % chunk_size` must be 0 for both chunk sizes.
    chunk_attention: bool
    q_chunk_size: Optional[int]
    kv_chunk_size: Optional[int]

    n_layer: int
    embd_dim: int
    embd_pdrop: float  # Dropout probability.

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
            # Chunking.
            chunk_attention=False,
            q_chunk_size=None,
            kv_chunk_size=None,
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


class MultiheadedCausalSelfAttention(nn.Module):
    """A simple causal self-attention module, with standard K/Q/V + output projection
    matrices."""

    # TODO: we could probably reduce the amount of duplication between here and
    # GPTConfig.
    n_head: int
    resid_pdrop: float
    attn_pdrop: float
    embd_dim: int  # Not actually needed, just used for assertion on input shape.

    chunk_attention: bool
    q_chunk_size: Optional[int]
    kv_chunk_size: Optional[int]

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

        def causal_self_attention(
            k: jnp.ndarray,
            q: jnp.ndarray,
            v: jnp.ndarray,
            dropout_key: jax.random.KeyArray,
        ):
            """Helper for running causal self-attention, with dropout."""
            assert k.shape == q.shape == v.shape == (T, head_size)
            if self.chunk_attention:
                # Chunked self-attention. Slower, but requires less memory.
                assert self.q_chunk_size is not None
                assert self.kv_chunk_size is not None
                return attention.causal_self_attention_chunked(
                    k,
                    q,
                    v,
                    dropout_key=dropout_key,
                    pdrop=self.attn_pdrop,
                    deterministic=deterministic,
                    q_chunk_size=self.q_chunk_size,
                    kv_chunk_size=self.kv_chunk_size,
                )
            else:
                # Naive self-attention. Quadratic memory requirement.
                return attention.causal_self_attention_naive(
                    k,
                    q,
                    v,
                    dropout_key=dropout_key,
                    pdrop=self.attn_pdrop,
                    deterministic=deterministic,
                )

        # Vectorize each attention head + the batch axis, passing each a unique dropout
        # key. TODO: it should be straightforward to fold the vmapped axes into the
        # self-attention einsums; this would eliminate all of the PRNG key splitting.
        if deterministic:
            # In deterministic mode, the dropout PRNG key is not actually used, so we
            # don't require that it's passed in.
            dropout_key = jax.random.PRNGKey(0)
        else:
            dropout_key = self.make_rng("dropout")
        dropout_key = jax.random.split(dropout_key, B)
        dropout_key = jax.vmap(lambda k: jax.random.split(k, self.n_head))(dropout_key)
        assert dropout_key.shape[:2] == (B, self.n_head)

        y = jax.vmap(  # vmap over each head.
            jax.vmap(causal_self_attention),  # vmap over batch axis.
            in_axes=(2, 2, 2, 1),
            out_axes=2,
        )(k, q, v, dropout_key)
        assert y.shape == (B, T, self.n_head, head_size)

        # Output projection, dropout.
        y = y.reshape((B, T, C))
        y = DenseWithInit(features=C)(y)
        y = nn.Dropout(rate=self.resid_pdrop, deterministic=deterministic)(y)
        assert y.shape == x.shape == (B, T, C)

        return y


class GPTBlock(nn.Module):
    """An unassuming GPT block."""

    config: GPTConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:  # type: ignore
        x = x + MultiheadedCausalSelfAttention(
            # TODO: we could probably reduce the boilerplate here.
            n_head=self.config.n_head,
            resid_pdrop=self.config.resid_pdrop,
            attn_pdrop=self.config.attn_pdrop,
            embd_dim=self.config.embd_dim,
            chunk_attention=self.config.chunk_attention,
            q_chunk_size=self.config.q_chunk_size,
            kv_chunk_size=self.config.kv_chunk_size,
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
