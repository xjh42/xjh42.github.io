---
title: 'DLSys: Manual Neural Networks'
date: 2025-02-24 13:05:58
tags:
- nn_basics
categories:
- dlsys

pin: false
math: true
---

In this post, I'll introduce nerual network basics.

# Nonlinear hypothesis classes

Recall that we needed a hypothesis function to map inputs in $R^n$ to outputs (class logits) in $R^k$, so we initially used the linear hypothesis class.
$$
h_\theta(x) = \theta^Tx, \theta \in R^{n \times k}
$$
This classifier essentially forms k linear functions of the input and then predicts the class with the largest value: equivalent to partitioning the input into k linear regions corresponding to each class.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240924131137657.png" alt="image-20240924131137657" style="zoom:33%;" />

What if we have data that cannot be separated by a set of linear regions? We want some way to separate these points via a nonlinear set of class boundaries.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240924131225307.png" alt="image-20240924131225307" style="zoom:33%;" />

We can apply a linear classifier to **some (potentially higher-dimensional) features of the data**.
$$
h_\theta(x) = \theta^T\phi(x), \theta \in R^{d \times k}, \phi: R^n \to R^d
$$


## Nonlinear Feature

How can we create the feature function $\phi$? 

- Through manual engineering of features relevant to the problem (the “old” way of doing machine learning) 
- In a way that itself is learned from data (the “new” way of doing ML)

First take: what if we just again use a linear function for $\phi $?
$$
\phi(x) = W^Tx
$$


Doesn’t work, because it is just equivalent to another linear classifier:
$$
h_\theta(x) = \theta^T\phi(x)=\theta^TW^Tx = \hat{\theta}x
$$
We can add nonlinear function the result like:
$$
\phi(x) = \sigma(W^Tx)
$$
where $W \in R^{n \times d}$, and $ \phi: R^d \to R^d $ is essentially any nonlinear function.

Example: let W be a (fixed) matrix of random Gaussian samples, and let $ \phi $ be the cosine function ⟹ “random Fourier features” (work great for many problems) But maybe we want to train $ W $ to minimize loss as well? Or maybe we want to compose multiple features together? 

# Neural networks

A neural network refers to a particular type of hypothesis class, consisting of **multiple, parameterized differentiable functions (a.k.a. “layers”) composed together in any manner to form the output.**

## The “two layer” neural network

We can begin with the simplest form of neural network, basically just the nonlinear features proposed earlier, but where both sets of weights are learnable parameters.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240924132433576.png" alt="image-20240924132433576" style="zoom:33%;" />
$$
h_\theta(x)=W_2^T\sigma(W_1^Tx), \theta = \{W_1 \in R^{n \times d},W_2 \in R^{d \times k } \}
$$


where $\sigma: R \to R$ is a nonlinear function applied elementwise to the vector (e.g. sigmoid, ReLU). Written in batch matrix form:
$$
h_\theta(X) = \sigma(XW_1)W_2
$$


# Backpropagation

Recall that neural networks just specify one of the “three” ingredients” of a machine learning algorithm. We also need: 

- Loss function: still cross entropy loss, like last time 
- Optimization procedure: still SGD, like last time

In other words, we still want to solve the optimization problem:
$$
min_{\theta} \frac{1}{m} \sum_{i=1}^{m} l_{ce}(h_{\theta}(x^{(i)}, y^{(i)}))
$$
using SGD, just with $h_{\theta}(x)$ now being a neural network. Requires computing the gradients  for each element of $\theta $

Some slides which do not need to remember:

![image-20241224093604458](https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20241224093604458.png)

![image-20241224093627447](https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20241224093627447.png)

![image-20241224093800729](https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20241224093800729.png)
