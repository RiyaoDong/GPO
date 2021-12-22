# GPO

Implementation of Gradual Projection Operators for for Sparse Reparameterization, which is proposed in [Learning Soft Threshold for Sparse Reparameterization
using Gradual Projection Operators](under review).

## Requirements
```
Python 3, PyTorch >= 1.1.0
```

## Datasets preparation

We conduct our experiments on four popular datasets, e.g. CIFAR-10, CIFAR-100, Tiny-Imagenet, Imagenet. For your convenience, we provide getTinyim.sh script to 
download and process Tiny-Imagenet dataset. You can prepare the other datasets yourself.
```
bash getTinyim.sh
```

## Training a ResNet on popular datasets (e.g. CIFAR-100, Tiny-Imagenet) with Gradual Projection Operators

The main_gpu.sh can be used to train a ResNet-32 on CIFAR-100 with Gradual Projection Operators. You can change the arch and dataset in main_gpu.sh accordingly
to conduct other experiments.
```
bash main_gpu.sh
```

## Contact

Feel free to discuss papers/code with us through issues/emails!

wdecen@foxmail.com


