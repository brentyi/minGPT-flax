# minGPT-flax

A basic transformer implementation, for seq2seq modeling in Flax/JAX. Written
for educational purposes :school:.

Also includes some bells and whistles:

- Data parallelism. By default, we train on all available GPUs. This can
  massively speed up training, even on smaller batch sizes.
- Chunked self-attention, adapted from the approach described by Markus Rabe and
  Charles Staats [1]. This makes a small runtime tradeoff to avoid the quadratic
  memory constraint of standard self-attention.

[1] https://arxiv.org/pdf/2112.05682v2.pdf

## Usage

Install:

```
pip install -r requirements.txt
```

Train:

```
$ python train_char.py --help

usage: train_char.py [-h] --dataset-path PATH [--experiment-name STR] [--restore-checkpoint] [--max-epochs INT]
                     [--minibatch-size INT] [--block-size INT] [--gpt-config.vocab-size INT] [--gpt-config.block-size INT]
                     [--gpt-config.n-head INT] [--gpt-config.resid-pdrop FLOAT] [--gpt-config.attn-pdrop FLOAT]
                     [--gpt-config.chunk-attention] [--gpt-config.q-chunk-size INT] [--gpt-config.kv-chunk-size INT]
                     [--gpt-config.n-layer INT] [--gpt-config.embd-dim INT] [--gpt-config.embd-pdrop FLOAT]
                     [--optimizer-config.learning-rate FLOAT] [--optimizer-config.no-lr-decay]
                     [--optimizer-config.adam-b1 FLOAT] [--optimizer-config.adam-b2 FLOAT]
                     [--optimizer-config.warmup-tokens INT] [--optimizer-config.final-tokens INT]
                     [--optimizer-config.weight-decay FLOAT] [--optimizer-config.grad-norm-clip FLOAT]

required arguments:
  --dataset-path PATH   Path to a text file, to be loaded for training. Needs to fit in memory.

optional arguments:
  -h, --help            show this help message and exit
  --experiment-name STR
                        (default: char_2022-01-07-18:01:54)
  --restore-checkpoint
  --max-epochs INT      (default: 1000)
  --minibatch-size INT  (default: 128)
  --block-size INT      (default: 128)
  --gpt-config.vocab-size INT
                        (default: 256)
  --gpt-config.block-size INT
                        The history/context length of our sequence model. (default: 128)
  --gpt-config.n-head INT
                        Output size for multi-headed self-attention. (default: 8)
  --gpt-config.resid-pdrop FLOAT
                        Dropout probability. (default: 0.1)
  --gpt-config.attn-pdrop FLOAT
                        Dropout probability. (default: 0.1)
  --gpt-config.chunk-attention
                        Enable attention chunking to trade runtime for memory efficiency. We implement an
                        approach similar to the algorithm presented here:
                        https://arxiv.org/pdf/2112.05682v2.pdf

                        If chunking is enabled, both q_chunk_size and kv_chunk_size must be set.
                        Note that `block_size % chunk_size` must be 0 for both chunk sizes.
  --gpt-config.q-chunk-size INT
                        (default: None)
  --gpt-config.kv-chunk-size INT
                        (default: None)
  --gpt-config.n-layer INT
                        (default: 8)
  --gpt-config.embd-dim INT
                        (default: 512)
  --gpt-config.embd-pdrop FLOAT
                        Dropout probability. (default: 0.1)
  --optimizer-config.learning-rate FLOAT
                        (default: 0.0006)
  --optimizer-config.no-lr-decay
                        If decay is enabled, we use cosine annealing.
  --optimizer-config.adam-b1 FLOAT
                        (default: 0.9)
  --optimizer-config.adam-b2 FLOAT
                        (default: 0.95)
  --optimizer-config.warmup-tokens INT
                        Tokens before reaching full learning rate. (default: 10240)
  --optimizer-config.final-tokens INT
                        At what point we reach 10% of original LR (default: 2560)
  --optimizer-config.weight-decay FLOAT
                        L2 regularization coefficient. (default: 0.1)
  --optimizer-config.grad-norm-clip FLOAT
                        (default: 1.0)
```

As an example, to train with self-attention chunk sizes of 64:

```
$ python train_char.py --dataset-path ./some_text_file --gpt-config.chunk-attention --gpt-config.q-chunk-size 64 --gpt-config.kv-chunk-size 64
```

The training script will attempt to use all available GPUs;
`CUDA_VISIBLE_DEVICES` may be helpful if this is undesired.

Eval (sampling):

```
$ python eval_char.py

usage: eval_char.py [-h] --experiment-name STR [--sample-steps INT] [--sample-from-top-k INT]

required arguments:
  --experiment-name STR

optional arguments:
  -h, --help            show this help message and exit
  --sample-steps INT    (default: 500)
  --sample-from-top-k INT
```

## Links

Third-party:

- The core model implementation details are based off of
  [karpathy/minGPT](https://github.com/mgrankin/minGPT) (PyTorch).
- The learning rate scheduler is adapted from
  [mgrankin/minGPT](https://github.com/mgrankin/minGPT) (Haiku).
- [matthias-wright/flaxmodels](https://github.com/matthias-wright/flaxmodels)
  also has pretrained models implemented using Flax.

This repo also serves as a testbed for a few "core infrastructure" libraries
that I've been working on, including:

- [fifteen](https://github.com/brentyi/fifteen), which contains utilities for
  training: data loading, experiment management, etc.
- [dcargs](https://github.com/brentyi/dcargs), which is used for unifying
  experiment configuration with type-safe argument parsing.
- [jax_dataclasses](https://github.com/brentyi/jax_dataclasses), which is used
  to construct type-safe PyTree structures.

## To-do list

- [x] Model
  - [x] Config classes
  - [x] Masked self-attention
    - [x] Naive version.
    - [x] Chunked version. https://arxiv.org/pdf/2112.05682v2.pdf
  - [x] Transformer + transformer blocks
- [x] Training boilerplate
  - [x] Minimal training loop
  - [x] Weight decay masking
  - [x] Learning rate scheduling
  - [x] Tensorboard logging
  - [x] Checkpointing
  - [x] Multi-GPU support
- [ ] Demos
  - [x] Character-level language model
    - [x] Training script
      ```bash
      python train_char.py --help
      ```
    - [x] Sampling/rollout demo
      ```bash
      python eval_char.py --help
      ```
  - [ ] BPE-based language model
- [ ] Tangentially related reach goals
  - [ ] Vision transformer
  - [ ] Transformer w/ Perceiver IO-style latent vectors?
  - [ ] Weight loading from OpenAI's released model?
