---
title: 'DLSys: Automatic Differentiation'
date: 2025-02-07 09:19:13
tags:
- autodiff
categories:
- dlsys

pin: false
math: true
---

In this post, I'll introduce the core of deep learning framework: **automatic differentiation**. In the first part, I'll introduce the thoery for automatic differentiation--mostly about **reverse automatic differentiation**. In the second part, I'll introduce how to implement a basic deep learning framework based on autodiff.

# Theory

So why calculating differentation is important in deep learning? Let's first look at the three basic elements in deep learning:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250210093510526.png" alt="image-20250210093510526" style="zoom: 50%;" />

We can see computing the loss function gradient with respect to hypothesis class parameters is the most common operation in machine learning.

## Common Differentiation Methods

### Numerical differentiation

We can directly compute the partial gradient by definition:
$$
\frac{\partial f(\theta)}{\partial \theta_i} = \lim_{\epsilon \rightarrow 0} \frac{f(\theta + \epsilon e_i) - f(\theta)}{\epsilon}
$$


Where $e_i$ is a vector: $[0, 0, ..., 1, 0, ..., 0]$ which means in `i`th position its value is 1.

A more numerically accurate way to approximate the gradient is:
$$
\frac{\partial f(\theta)}{\partial \theta_i} = \frac{f(\theta + \epsilon e_i) - f(\theta - \epsilon e_i)}{2 \epsilon} + o(\epsilon^2)
$$
However, this method to compute differentiation suffers from**numerical error** and is **less efficient to compute**(need calculate many times based on the size of $\theta$).

However, numerical differentiation is a powerful tool to **check an implement of an automatic differentiation algorithm in unit test cases**. We'll use the following equation to calculate the approximate result and compare to the framework result:
$$
\delta^T\nabla_\theta f(\theta) = \frac{f(\theta + \epsilon \delta) - f(\theta - \epsilon \delta)}{2 \epsilon} + o(\epsilon^2)
$$


We'll pick 𝛿 from unit ball, check the above invariance.

### Symbolic Differentiation

We can also use the formula from calculus course:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250210095405463.png" alt="image-20250210095405463" style="zoom:50%;" />

Use naive synbolic differentiation formula is costly, for example:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250210095526278.png" alt="image-20250210095526278" style="zoom:33%;" />

It'll cost 𝑛(𝑛 − 2) multiplies to compute all partial gradients for $\theta$

## Automatic Differentiation

Let's first introduce a definition: **computational graph** which convert a computation formula to a DAG(direct acylic graph). For example, we can convert $y = f(𝑥1, 𝑥2) = ln(𝑥1) + 𝑥1𝑥2 − sin(𝑥2)$ to following graph:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250210100111541.png" alt="image-20250210100111541" style="zoom:50%;" />

In computational graph, each node represent an (intermediate) value in the computation. Edges present input output relations. 

So the compuataion of a formula converts to the forward evaluation trace of the graph:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250210100327886.png" alt="image-20250210100327886" style="zoom:50%;" />

### Forward mode automatic differentiation

In forward mode automatic differentiation algorithm, we define $\dot{v_i}$ as:
$$
\dot{v_i} = \frac{\partial v_i}{\partial x_i}
$$
The we compute the $\dot{v_i}$ iteratively in the **forward topological order** of the computational graph. We use the computation of $y = f(𝑥1, 𝑥2) = ln(𝑥1) + 𝑥1𝑥2 − sin(𝑥2)$ as example:

- The comptational graph:

  <img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250210101101279.png" alt="image-20250210101101279" style="zoom:50%;" />

- Forward evaluation trace

  <img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250210101131656.png" alt="image-20250210101131656" style="zoom:50%;" />

- Forward AD trace

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250210101248975.png" alt="image-20250210101248975" style="zoom:50%;" />

In Forward AD, for $f: R^n\rightarrow R^k$, we need 𝑛 forward AD passes to get the gradient with respect to each input. In deep learning, we mostly care about the cases where 𝑘 is small and large 𝑛. In order to resolve the problem efficiently, we need to use another kind of AD.

### Reverse mode automatic differentiation

In reverse mode automatic differentiation, we define $\bar{v_i}$ as:
$$
\bar{v_i} = \frac{\partial y}{\partial v_i}
$$
We can then compute the $\bar{v_i}$ iteratively in the **reverse topological order** of the computational graph. We use the computation of $y = f(𝑥1, 𝑥2) = ln(𝑥1) + 𝑥1𝑥2 − sin(𝑥2)$ as example:

- The comptational graph:

  <img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250210101101279.png" alt="image-20250210101101279" style="zoom:50%;" />

- Forward evaluation trace

  <img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250210101131656.png" alt="image-20250210101131656" style="zoom:50%;" />

  - Reverse AD evaluation trace

  <img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250210102212166.png" alt="image-20250210102212166" style="zoom:50%;" />

In In reverse mode automatic differentiation,  we need compute derivation for the multiple pathway. For example, $v_1$ s being used in multiple pathways ($v_2$ and $v_3$).

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250210124140952.png" alt="image-20250210124140952" style="zoom:50%;" />

$y$ can be written in the form of $y = f(v2, v3)$

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250210124243642.png" alt="image-20250210124243642" style="zoom:50%;" />

We define partial adjoint $\bar{v_{i \rightarrow j}} = \bar{v_j} \frac{\partial v_j}{ \partial v_i}$, for each input output node pair $𝑖$ and $j$ , then we can compute $\bar{v_i}$ as:
$$
\bar{v_i} = \sum_{j \in next{i}}{\bar{v_{i \rightarrow j}}}
$$


We can compute partial adjoints separately then sum them together.

In practice, we can implement reverse ad algorithm as follows:

```python
def gradient(out):
  node_to_grad = {out: [1]} # Dictionary that records a list of partial adjoints of each node
  for i n reverse_topo_order(out):
    adj_derivation = sum(node_to_grad[i]) # Sum up partial adjoints
    
    for k in input(i):
      compute partial_k2i = adj_derivation * dev_k2i
      append partial_k2i to node_to_grad[k] # “Propagates” partial adjoint to its input
      
  return sum of node_to_grad[input] 
```

An example $y = (exp(x) + 1)exp(x)$ is showed as follow:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250210130411215.png" alt="image-20250210130411215" style="zoom:50%;" />

We build another computational graph to calculate the gradient.

Compare to backprop, we can find that :

- Reverse mode ad is effient to calculate gradient of gradient 
- Reverse mode ad is easy to optimize by fuse nodes...

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250210130713960.png" alt="image-20250210130713960" style="zoom:40%;" />

Currently, we only apply reverse mod ad on scalar variable. Now we can expend the algorithm to tensor variables.

Let's see how to use the algorithm on a simple equation: $y = f(XW)$, the comptational graph is:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250210131048881.png" alt="image-20250210131048881" style="zoom:50%;" />

- Forwad evaluation trace:
  $$
  Z_{ij} = \sum_{k}{X_{ik}W_{kj}}
  $$

  $$
  v = f(Z)
  $$

  

   We can write them in matric form:
$$
Z =XW
$$

$$
v = f(Z)
$$



- Define adjoint for tensor values $\bar{Z}$:
  $$
  \bar Z = \begin{bmatrix} 
  \frac{\partial y}{\partial Z_{1,1}} & ... & \frac{\partial y}{\partial Z_{1,n}} \\
  ... & ... & ... \\
  \frac{\partial y}{\partial Z_{n,1}} & ... & \frac{\partial y}{\partial Z_{n,n}}
  \end{bmatrix}
  $$

- Reverse evaluation in scalar form for $\bar{X}$

$$
\bar X_{i.k} = \sum_j{\bar Z_{i, j} \frac{\partial Z_{i, j}}{\partial X_{i.k}}} 
= \sum_j {W_{k, j}, \bar Z_{i, j}}
$$

   We can also write it in matrix form:
$$
\bar X = \bar Z W^T
$$
We can also use reverse mode ad on data structures:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250210132920474.png" alt="image-20250210132920474" style="zoom:50%;" />

# Implementation

