# minGPT-flax

A minimal transformer implementation with data parallelism, for seq2seq modeling
in Flax/JAX. Written for educational purposes :school:

## Usage

Install:

```
pip install -r requirements.txt
```

Train:

```
usage: train_char.py [-h] --dataset-path PATH [--experiment-name STR] [--restore-checkpoint] [--max-epochs INT] [--minibatch-size INT] [--block-size INT] [--gpt-config.vocab-size INT] [--gpt-config.block-size INT] [--gpt-config.n-head INT] [--gpt-config.resid-pdrop FLOAT]
                     [--gpt-config.attn-pdrop FLOAT] [--gpt-config.n-layer INT] [--gpt-config.embd-dim INT] [--gpt-config.embd-pdrop FLOAT] [--optimizer-config.learning-rate FLOAT] [--optimizer-config.no-lr-decay] [--optimizer-config.adam-b1 FLOAT]
                     [--optimizer-config.adam-b2 FLOAT] [--optimizer-config.warmup-tokens INT] [--optimizer-config.final-tokens INT] [--optimizer-config.weight-decay FLOAT] [--optimizer-config.grad-norm-clip FLOAT]

required arguments:
  --dataset-path PATH   Path to a text file, to be loaded for training. Needs to fit in memory.

optional arguments:
  -h, --help            show this help message and exit
  --experiment-name STR
                        (default: char_2022-01-02-03:46:52)
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

- The core model implementation details are based off of
  [karpathy/minGPT](https://github.com/mgrankin/minGPT) (PyTorch).
- The learning rate scheduler is adapted from
  [mgrankin/minGPT](https://github.com/mgrankin/minGPT) (Haiku).
- Experiment management is borrowed from
  [this project](https://github.com/brentyi/dfgo)'s infrastructure.
- [matthias-wright/flaxmodels](https://github.com/matthias-wright/flaxmodels)
  also has pretrained models implemented using Flax.

## To-do list

- [x] Model
  - [x] Config classes
  - [x] Masked self-attention
    - [x] Version written from scratch
    - [ ] Version using `flax.linen.attention`
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
