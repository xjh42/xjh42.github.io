---
title: 'DLSys: ML Refresher / Softmax Regression'
date: 2025-02-01 09:07:04
tags:
- ml_basics
categories:
- dlsys

pin: false
math: true
---

In this post, I'll introduce the machine learning basics.

# Machine Learning Basics

Suppose you want to write a program that will classify handwritten drawing of digits into their appropriate category: 0,1,…,9. You could, think hard about the nature of digits, try to determine the logic of what indicates what kind of digit, and write a program to codify this logic (Despite being a reasonable coder, I don’t think I could do this very well)

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240924091119545.png" alt="image-20240924091119545" style="zoom: 45%;" />

The (supervised) **ML approach**: collect a training set of images with known labels and feed these into a machine learning algorithm, which will (if done well), automatically produce a “program” that solves this task.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240924091220614.png" alt="image-20240924091220614" style="zoom:50%;" />

Every machine learning algorithm consists of three different elements:

1. **The hypothesis class(the model)**: the “program structure”, parameterized via a set of parameters, that describes how we map inputs (e.g., images of digits) to outputs (e.g., class labels, or probabilities of different class labels)
2. **The loss function:** a function that specifies how “well” a given hypothesis (i.e., a choice of parameters) performs on the task of interest
3. **An optimization method:**  a procedure for determining a set of parameters that (approximately) minimize the sum of losses over the training set



# softmax regression

Softmax regression is a basic method to solve *k-class classification*  problem. Let's first define the problem:

## Multi-class classification setting

Let’s consider a k-class classification setting, where we have:

- **Training data**: $x^{(i)} \in R^n$, and $y^{(i)} \in {1,...,k}$ for $i \in {1,..,m}$
- 𝑛 = dimensionality of the input data
- 𝑘 = number of different classes / labels
- 𝑚 = number of points in the training set

Example: classification of 28x28 MNIST digits:

- 𝑛 = 28 ⋅ 28 = 784 
-  𝑘 = 10 
-  𝑚 = 60,000

## Three Element of Softmax Regression

### Linear hypothesis function

Our hypothesis function maps inputs $x \in R^n$  to 𝑘-dimensional vectors:
$$
h: R^n \to R^k
$$
where $h_i(x), i \in {1,..k}$ indicates some measure of “belief” in how much likely the label is to be class 𝑖 (i.e., “most likely” prediction is coordinate 𝑖 with largest $h_i(x)$.)

A **linear hypothesis function** uses a linear operator (i.e. matrix multiplication) for this transformation:
$$
h_\theta(x) = \theta^Tx
$$
for parameters $\theta \in R^{n \times k}$

Often more convenient (and this is how you want to code things for efficiency) to write the data and operations in **matrix batch form**:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240924093023427.png" alt="image-20240924093023427" style="zoom:33%;" />

Then the linear hypothesis applied to this batch can be written as:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240924093044172.png" alt="image-20240924093044172" style="zoom:33%;" />

### Loss Function

The simplest loss function to use in classification is just the **classification error**, i.e., whether the classifier makes a mistake a or not:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240924093324238.png" alt="image-20240924093324238" style="zoom:33%;" />

We typically use this loss function to assess the quality of classifiers.

Unfortunately, the error is a bad loss function to use for optimization, i.e., selecting the best parameters, because it is not differentiable.

### softmax / cross-entropy loss

Let’s convert the hypothesis function to a “probability” by **exponentiating and normalizing its entries** (to make them all positive and sum to one):
$$
z_i = p(label=i) = \frac{exp(h_i(x))}{\sum_{j=1}^kexp(h_j(x))} = norm(exp(h(y))
$$
Then let’s define a loss to be the (negative) log probability of the true class: this is called softmax or cross-entropy loss :
$$
l_{ce}(h(x),y) = -log(p(label=y)) = -h_y(x) + log\sum_{j=1}^kexp(h_j(x))
$$

### Softmax Regression Optimization Method: Stochastic Gradient Descent

The third ingredient of a machine learning algorithm is a method for solving the associated optimization problem, i.e., the problem of minimizing the average loss on the training set:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240924094501602.png" alt="image-20240924094501602" style="zoom:33%;" /> 

So how do we find Θ that solves this optimization problem?

For a matrix-input, scalar output function $f: R^{n \times k} \to R$  the gradient is defined as the matrix of partial derivatives:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240924094719645.png" alt="image-20240924094719645" style="zoom: 67%;" />

Gradient points in the direction that most increases 𝑓 (locally).

To minimize a function, the gradient descent algorithm proceeds by **iteratively taking steps in the direction of the negative gradient**:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240924094943710.png" alt="image-20240924094943710" style="zoom:50%;" />

where 𝛼 > 0 is a step size or learning rate:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240924095031145.png" alt="image-20240924095031145" style="zoom:50%;" />

If our objective (as is the case in machine learning) is the sum of individual losses, we don’t want to compute the gradient using all examples to make a single update to the parameters.

Instead, take many gradient steps each based upon a minibatch (small partition of the data), to make many parameter updates using a single “pass” over data

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240924095218315.png" alt="image-20240924095218315" style="zoom:50%;" />

So, how do we compute the gradient for the softmax objective?

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240924095421785.png" alt="image-20240924095421785" style="zoom:40%;" />

Let’s start by deriving the gradient of the softmax loss itself: for vector $h \in R^k$:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240924095509447.png" alt="image-20240924095509447" style="zoom:33%;" />

So, in vector form: $\bigtriangledown_hl_{ce}(h,y)=z-e_y$, where 𝑧 = norm(exp(h)).

Then, let’s compute the “derivative” of the loss:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240924100156250.png" alt="image-20240924100156250" style="zoom:33%;" />

So to make the dimensions work:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240924100246907.png" alt="image-20240924100246907" style="zoom:33%;" />

Same process works if we use “matrix batch” form of the loss:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240924100441453.png" alt="image-20240924100441453" style="zoom:33%;" />

## Softmax Regression All in One

Despite a fairly complex derivation, we should highly just how simple the final algorithm is:

- Repeat until parameters / loss converges 
  - Iterative over minibatches $X \in R^{B \times n}, y \in \{1,...,k\}^B$ of training set 
  -  Update the parameters $\theta = \theta - \frac{\alpha}{B}X^T(Z - I_y)$
