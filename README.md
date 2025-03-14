# TinyGrad Project


You like [pytorch](https://pytorch.org/)? You like [micrograd](https://github.com/karpathy/micrograd)? You love tinygrad! ❤️

tinygrad is something between [PyTorch](https://pytorch.org/) and [karpathy/micrograd](https://github.com/karpathy/micrograd). Maintained by [tiny corp](https://tinygrad.org/), this may not be the best deep learning framework, but it is a deep learning framework. Due to its extreme simplicity, it aims to be the easiest framework to add new accelerators to, with support for both inference and training. If XLA is CISC, tinygrad is RISC. tinygrad is still alpha software, but we raised some money to make it good. Someday, we will tape out chips.

## Install

While you can `pip install tinygrad`, we encourage you to install from source:

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
