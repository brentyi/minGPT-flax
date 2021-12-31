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
$ python train_char.py --help

usage: train_char.py [-h] --dataset-path PATH [--experiment-name STR] [--restore-checkpoint] [--max-epochs INT] [--batch-size INT]
                     [--block-size INT]

required arguments:
  --dataset-path PATH

optional arguments:
  -h, --help            show this help message and exit
  --experiment-name STR
                        (default: char_{timestamp})
  --restore-checkpoint  (default: False)
  --max-epochs INT      (default: 1000)
  --batch-size INT      (default: 128)
  --block-size INT      (default: 128)
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
