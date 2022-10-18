# TSCP2_pytorch
Time Series Change Point Detection based on Contrastive Predictive Coding pytorch implementation
[![issues](https://img.shields.io/github/issues/lee-seon-woo/TSCP2_pytorch)](https://github.com/hazdzz/STGCN/issues)
[![forks](https://img.shields.io/github/forks/lee-seon-woo/TSCP2_pytorch)](https://github.com/hazdzz/STGCN/network/members)
[![stars](https://img.shields.io/github/stars/lee-seon-woo/TSCP2_pytorch)](https://github.com/hazdzz/STGCN/stargazers)
[![License](https://img.shields.io/github/license/lee-seon-woo/TSCP2_pytorch)](./LICENSE)

* This repo covers an reference implementation for the following papers in PyTorch. Official Code and Paper as follow: [Tensorflow2: Repository](https://github.com/cruiseresearchgroup/TSCP2) and [TSCP2: Deldari, Shohreh and Smith, Daniel V. and Xue, Hao and Salim, Flora D.](https://arxiv.org/abs/2011.14097)
## Dataset
####  [Go to Page](https://github.com/OctoberChang/klcpd_code/tree/master/data)

## Differents of code between mine and author's
1. Pytorch Implementation
2. Add Early Stopping approach Using Trainer Class
3. Add Spatial Dropout approach
4. Offer a different set of hyperparameters using Hydra
5. Add Toy datasets
6. Add Attention Mechanism and Batchnorm

## Requirements
#### Python 3.9.13
To install requirements:
```console
pip install -r requirements.txt   or
conda create --name <env> --file conda_requirement.txt   or
conda create --name <env> --file environment.yml
```
### **Train**
```
python main.py
```

# Reference Papers
1. [Time Series Change Point Detection with Self-Supervised Contrastive Predictive Coding](https://doi.org/10.1145/3442381.3449903)
2. [Temporal Convolution Network](https://github.com/flrngel/TCN-with-attention)
3. [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
4. [Noise-contrastive estimation: A new estimation principle for unnormalized statistical models](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)
5. [InfoNCE_Loss](https://github.com/loeweX/Greedy_InfoMax/)
6. [Greedy_InfoMax](https://github.com/loeweX/Greedy_InfoMax/)
7. [Kernel Change-point Detection with Auxiliary Deep Generative Models](https://openreview.net/forum?id=r1GbfhRqF7)