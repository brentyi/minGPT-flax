import functools
import math
from typing import Tuple

import jax
import jax_dataclasses as jdc
from flax import linen as nn
from jax import numpy as jnp


def _dropout(x: jnp.ndarray, rate: float, key: jax.random.KeyArray) -> jnp.ndarray:
    """Functional dropout implementation. In contrast to the flax.linen module, this can
    be used inside of standard JAX function transforms.

    Note that we could also use the lifted transforms provided by Flax, but this
    is more general."""
    keep_prob = 1.0 - rate
    mask = jax.random.bernoulli(key, p=keep_prob, shape=x.shape)
    out = jnp.where(mask, x / keep_prob, 0.0)
    assert out.shape == x.shape
    return out


def causal_self_attention_naive(
    k: jnp.ndarray,
    q: jnp.ndarray,
    v: jnp.ndarray,
    *,
    dropout_key: jax.random.KeyArray,
    pdrop: float,
    deterministic: bool,
) -> jnp.ndarray:
    """Simple causal self-attention implementation. This is fast, but requires quadratic
    memory.

    k, q, and v should all be arrays of shape (token count, channel/head size). Support for
    batch axes or multiple heads is simple with a vmap."""

    # Check shapes. Note that this is actually a generalized attention implementation,
    # but our asserts assume self-attention and are overly restrictive!
    T, C = k.shape
    assert k.shape == q.shape == v.shape

    # For computing QK^T:
    #     Inputs are: (T, C), (T, C)
    #     Desired output is: (T, T)
    # Dividing by square root of the head size results in better gradients for
    # softmax (more values on the locally linear area).
    att = jnp.einsum("kc,qc->qk", k, q) / math.sqrt(C)
    assert att.shape == (T, T)

    # Create and apply a causal mask.
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    att = jnp.where(mask, att, -jnp.inf)
    assert att.shape == (T, T)

    # Softmax over last axis.
    att = nn.softmax(att, axis=-1)
    assert att.shape == (T, T)

    # Dropout.
    if not deterministic:
        att = _dropout(att, rate=pdrop, key=dropout_key)

    # For computing softmax(QK^T/sqrt(head_size)).
    #     Inputs are: (T, T), (T, C)
    #     Desired output is: (T, C)
    # note that softmax was applied to the final T dimension of `att`!
    out = jnp.einsum("qk,kc->qc", att, v)
    assert out.shape == (T, C)
    return out


def causal_self_attention_chunked(
    k: jnp.ndarray,
    q: jnp.ndarray,
    v: jnp.ndarray,
    *,
    dropout_key: jax.random.KeyArray,
    pdrop: float,
    deterministic: bool,
    q_chunk_size: int,
    kv_chunk_size: int,
) -> jnp.ndarray:
    """Self-attention with chunking for both the queries and the keys. This may be
    slightly slower than the naive implementation, but requires significantly less
    memory. Based on the approach described in [1], but modified for masking + dropout.


    [1] https://arxiv.org/pdf/2112.05682v2.pdf

    Code here could likely be refactored.

    k, q, and v should all be arrays of shape (token count, channel/head size). A
    batch axis can be added with a vmap."""

    # Clip chunk sizes.
    q_chunk_size = min(q_chunk_size, q.shape[0])
    kv_chunk_size = min(kv_chunk_size, k.shape[0])

    # Check shapes.
    T, C = k.shape
    assert k.shape == q.shape == v.shape
    assert T % q_chunk_size == 0

    # Division by square root of the head size results in better gradients for softmax
    # (more values on the locally linear area).
    q = q / math.sqrt(C)

    def chunk_scanner(
        carry_q_start_idx: int, x_dropout_key: jax.random.KeyArray
    ) -> Tuple[int, jnp.ndarray]:
        q_chunk = jax.lax.dynamic_slice(
            q,
            start_indices=(carry_q_start_idx, 0),
            slice_sizes=(q_chunk_size, C),
        )
        out = _causal_self_attention_chunked_kv(
            k,
            q_chunk,
            v,
            q_start_idx=carry_q_start_idx,
            dropout_key=x_dropout_key,
            pdrop=pdrop,
            deterministic=deterministic,
            kv_chunk_size=kv_chunk_size,
        )
        return (carry_q_start_idx + q_chunk_size, out)

    _, out = jax.lax.scan(
        f=chunk_scanner,
        init=0,
        xs=jax.random.split(dropout_key, num=T // q_chunk_size),
    )
    assert out.shape == (T // q_chunk_size, q_chunk_size, C)
    return out.reshape((T, C))


@jdc.pytree_dataclass
class _ChunkSummary:
    exp_values: jnp.ndarray
    exp_weights_summed: jnp.ndarray
    max_score: jnp.ndarray


def _causal_self_attention_chunked_kv(
    k: jnp.ndarray,
    q: jnp.ndarray,
    v: jnp.ndarray,
    *,
    q_start_idx: int,
    dropout_key: jax.random.KeyArray,
    pdrop: float,
    deterministic: bool,
    kv_chunk_size: int,
) -> jnp.ndarray:
    """Self-attention with chunking for only the key/value pairs."""

    assert len(k.shape) == len(q.shape) == len(v.shape) == 2
    assert k.shape[-1] == q.shape[-1] == v.shape[-1]
    assert k.shape == v.shape, "Key and value shapes must match."
    assert k.shape[0] % kv_chunk_size == 0
    C = k.shape[-1]

    def compute_chunk(inputs: Tuple[int, jax.random.KeyArray]) -> _ChunkSummary:
        kv_start_idx, chunk_dropout_key = inputs
        k_chunk = jax.lax.dynamic_slice(
            k, start_indices=(kv_start_idx, 0), slice_sizes=(kv_chunk_size, C)
        )
        v_chunk = jax.lax.dynamic_slice(
            v, start_indices=(kv_start_idx, 0), slice_sizes=(kv_chunk_size, C)
        )
        return summarize_chunk(
            k_chunk,
            q,
            v_chunk,
            kv_start_idx=kv_start_idx,
            chunk_dropout_key=chunk_dropout_key,
        )

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(
        k: jnp.ndarray,
        q: jnp.ndarray,
        v: jnp.ndarray,
        *,
        kv_start_idx: int,
        chunk_dropout_key: jax.random.KeyArray,
    ) -> _ChunkSummary:
        assert len(k.shape) == len(q.shape) == len(v.shape) == 2
        assert k.shape[-1] == q.shape[-1] == v.shape[-1]
        assert k.shape == v.shape, "Key and value shapes must match."
        q_count = q.shape[0]
        kv_count = k.shape[0]

        C = v.shape[-1]

        attn_weights = jnp.einsum("kc,qc->qk", k, q)

        # Causal mask.
        mask = (q_start_idx + jnp.arange(q.shape[0]))[:, None] >= (
            kv_start_idx + jnp.arange(k.shape[0])
        )[None, :]
        assert mask.shape == attn_weights.shape == (q_count, kv_count)
        attn_weights = jnp.where(mask, attn_weights, -jnp.inf)

        max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        max_score = jax.lax.stop_gradient(max_score)

        # Avoid NaNs with a shim. (because `-jnp.inf + jnp.inf` => NaN)
        max_score_shimmed = jax.lax.stop_gradient(
            jnp.where(
                jnp.isneginf(max_score),
                0.0,  # Any real number should do here.
                max_score,
            )
        )
        exp_weights = jnp.exp(attn_weights - max_score_shimmed)

        # Dropout.
        if not deterministic:
            exp_weights = _dropout(exp_weights, rate=pdrop, key=chunk_dropout_key)

        # Compute values.
        exp_values = jnp.einsum("kc,qk->qc", v, exp_weights)
        exp_weights_summed = exp_weights.sum(axis=-1, keepdims=True)

        assert exp_values.shape == (q_count, C)
        assert exp_weights.shape == (q_count, kv_count)
        assert exp_weights_summed.shape == (q_count, 1)
        assert max_score.shape == (q_count, 1)

        return _ChunkSummary(
            # Weighted sum of values for each query.
            exp_values=exp_values,
            # Summed weights for each query in the chunk.
            exp_weights_summed=exp_weights_summed,
            # Max score for each query in the chunk.
            max_score=max_score,
        )

    dropout_keys = jax.random.split(dropout_key, k.shape[0] // kv_chunk_size)
    chunks = jax.lax.map(
        compute_chunk,
        (jnp.arange(0, k.shape[0], kv_chunk_size), dropout_keys),
    )

    chunk_count = k.shape[0] // kv_chunk_size
    q_count = q.shape[0]

    assert chunks.exp_values.shape == (chunk_count, q_count, C)
    assert chunks.exp_weights_summed.shape == (chunk_count, q_count, 1)
    assert chunks.max_score.shape == (chunk_count, q_count, 1)

    global_max_score = jnp.max(chunks.max_score, axis=0, keepdims=True)
    assert global_max_score.shape == (1, q_count, 1)

    max_score_diffs = jnp.exp(chunks.max_score - global_max_score)
    reweighted_values = max_score_diffs * chunks.exp_values
    reweighted_weights_summed = max_score_diffs * chunks.exp_weights_summed
    assert reweighted_values.shape == (chunk_count, q_count, C)
    assert reweighted_weights_summed.shape == (chunk_count, q_count, 1)

    out = jnp.sum(reweighted_values, axis=0) / (
        # Due to dropout, the summed weights will sometimes (very rarely) total to 0. We
        # add a small epsilonto prevent NaNs.
        #
        # In general this indicates that adding dropout to the the chunked
        # self-attention implementation may lead to operations that are not numerically
        # stable; should be revisited for any real applications.
        jnp.sum(reweighted_weights_summed, axis=0)
        + 1e-7
    )
    assert out.shape == (q_count, C)
    return out


def _check():
    """Check that our two self-attention implementations match."""

    import numpy as onp

    T = 1024
    C = 64

    onp.random.seed(3)
    k = onp.random.randn(T, C)
    q = onp.random.randn(T, C)
    v = onp.random.randn(T, C)

    dropout_key = jax.random.PRNGKey(0)

    chunked = jax.jit(
        lambda k, q, v: causal_self_attention_chunked(
            k,
            q,
            v,
            dropout_key=dropout_key,
            pdrop=0,
            deterministic=True,
            q_chunk_size=2,
            kv_chunk_size=2,
        )
    )(k, q, v)
    naive = jax.jit(
        lambda k, q, v: causal_self_attention_naive(
            k,
            q,
            v,
            dropout_key=dropout_key,
            pdrop=0,
            deterministic=True,
        )
    )(k, q, v)

    onp.testing.assert_allclose(
        chunked.flatten(),
        naive.flatten(),
        atol=1e-4,
        rtol=1e-4,
    )


if __name__ == "__main__":
    _check()
