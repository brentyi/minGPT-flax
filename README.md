# minGPT-flax

A minimal transformer implementation for seq2seq modeling in Flax/JAX. Written for educational purposes :school:


## Links

- The core model implementation details are based off of [karpathy/minGPT](https://github.com/mgrankin/minGPT) (PyTorch).
- The learning rate scheduler is adapted from [mgrankin/minGPT](https://github.com/mgrankin/minGPT) (Haiku).
- Experiment management is borrowed from [this project](https://github.com/brentyi/dfgo)'s infrastructure.
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
    - [ ] Vision transformer
    - [ ] Transformer w/ Perceiver IO-style latent vectors?
    - [ ] Weight loading from OpenAI's released model?
