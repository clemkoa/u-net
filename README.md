# U-net

A simple pytorch implementation of U-net, as described in the paper: https://arxiv.org/abs/1505.04597

This project is meant to be a dead-simple implementation of the model.
The only dependencies are pytorch, numpy and pillow.

The main differences with the paper are:
- no padding in the pooling, which makes handling dimensions easier
- no weight balancing in the softmax to deal with class inbalance

## Bibliography:
- https://github.com/milesial/Pytorch-UNet

## Installation

```
pip install torch numpy pillow
mkdir model
```
