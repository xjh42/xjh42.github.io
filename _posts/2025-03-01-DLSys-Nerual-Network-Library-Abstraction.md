---
title: 'DLSys: Nerual Network Library Abstraction'
date: 2025-03-01 09:50:18
tags:
- nn_lib
categories:
- dlsys

pin: false
math: true
---

Today I'll introduce the abstraction of nerual network which can help developer easily to build network application.

## Theory

## Programming Abstraction

Building nerual network from scratch is hard. We can see from the following picture that there are common pattern in nerual network building/traing:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250210093510526.png" alt="image-20250210093510526" style="zoom: 50%;" />

Therefore we can build a programming abstraction to fit the nerual network building/training process.The programming abstraction of a framework defines **the common ways to implement, extend and execute model computations**.

While the design choices may seem obvious after seeing them, it is useful to learn about the thought process, so that:

- We know why the abstractions are designed in this way.
- Learn lessons to design new abstractions

We'll study three useful framework to learn the design choices:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250226100223969.png" alt="image-20250226100223969" style="zoom:33%;" />

### Forward and backward layer interface

For eaxmple, **caffee 1** defines the forward computation and backward(gradient) operations:

```python
class Layer:
  def forward(bottom, top):
    pass
  def backward(top, propagate_down,bottom):
    pass
```

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250226100515220.png" alt="image-20250226100515220" style="zoom:50%;" />

Caffee 1 is design to use **backward propagation** to compute gradient. <u>So it's hard to build large network which is hard to compute the gradient.</u>

### Computational graph and declarative programming

Tensorflow 1.0 use **computational graph and reverse automatic differentiation** to build nerual network. It's much more easy to build large nerual network.

```python
import tensorflow as tf

v1 = tf.Variable()
v2 = tf.exp(v1)
v3 = v2 + 1
v4 = v2 * v3

sess = tf.Session()
value4 = sess.run(v4, feed_dict={v1: numpy.array([1]})
```

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250226101002492.png" alt="image-20250226101002492" style="zoom:40%;" />

However, the declarative language which use by tensorflow is hard to learn and write. And it's trick to define if/else in computation graph. We need use following weird code:

```
y = tf.cond(condition > 0, lambda: tf.matmul(x, W) + b, lambda: tf.matmul(x, W) - b)
```

### Imperative automatic differentiation

Like tensorflow 1.0, PyTorch use computational graph and reverse automatic differentiation to build network. However, it **executes computation as we construct the computational graph and allow easy mixing of python control flow and construction**.

```python
import needle as ndl
v1 = ndl.Tensor([1])
v2 = ndl.exp(v1)
v3 = v2 + 1
v4 = v2 * v3

# mix python control flow
if v4.numpy() > 0.5:
 v5 = v4 * 2
else:
 v5 = v4
v5.backward()
```

However, it'll be slow when you execute computation as we contruct the computation graph. So pytorch use `LAZY_MODE` to run computation when necessary.

## High level modular library components



We have three basic elements in machine learning: **The hypothesis class,The loss function,An optimization method**

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250226101939695.png" alt="image-20250226101939695" style="zoom:50%;" />

The question is how do they translate to modular components in code.

### Hypotheis Class

The hypothesis class is modular in nature. We can use modules to build a compound module. For example, a multi-layer residual net contains **3 residual block, 1 linear block and 1 softmax cross entropy**.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250226130132242.png" alt="image-20250226130132242" style="zoom:33%;" />

A residual block contains **linear blocks and relu blocks**.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250226130314159.png" alt="image-20250226130314159" style="zoom:33%;" />

So we can use `nn.Module` to compose things together.

- `nn.Module`:  input is tensor, output is tensor.

There are serveral things to consider:

- For given inputs, how to compute outputs
- How to get **the list of (trainable) parameters**
- Ways to **initialize the parameters**

### Loss Function

Just Like hypothesis class, loss function can be a special kind of module(tensor in, scalar out).

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250226130944472.png" alt="image-20250226130944472" style="zoom:33%;" />

To implement loss function, we need consider the following questions:

- How to compose multiple objective functions together? 
- What happens during inference time after training? 

### Optimizer

The model in fact is a list of parameters. The optimizer takes a list of weights from the model and perform steps of optimization. It may need to keep tracks of auxiliary states.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250226131318222.png" alt="image-20250226131318222" style="zoom:50%;" />

The common optimization algorithm is SGD, Momentum and Adam.

![image-20250226131419944](https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250226131419944.png)

### Initializer

Initialization strategy depends on the module being involved and the type of the parameter. Most neural network libraries have a set of common initialization routines

- **weights**: uniform, order of magnitude depends on input/output
- **bias**: zero
- **Running sum of variance**: one

Initialization can be folded into the construction phase of a `nn.module`.

### Data loader and preprocessing

We often need a class to handle data. And we often need perform preprocess to augmentation data to make nerual network more general.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250226131843377.png" alt="image-20250226131843377" style="zoom:33%;" />

We often preprocess (augment) the dataset by randomly shuffle and transform the input.Data augmentation can account for significant portion of prediction accuracy boost in deep learning models.Data loading and augmentation is also compositional in nature

### Put all in togather

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250226132000084.png" alt="image-20250226132000084" style="zoom:80%;" />

## Optimization

### Gradient descent

Let’s reconsider the generic gradient descent updates we described previously, now for a general function $f$, and writing iterate number $t$ explicitly:
$$
\theta_{t + 1} = \theta_t - \alpha \nabla_\theta f(\theta_t)
$$


where $\alpha > 0 $ is step size(learning rate), $\nabla_\theta f(\theta_t)$ is gradient evaluated at the parameters $\theta_t$. Gradient descent takes the “steepest descent direction” locally, but may oscillate over larger time scales. 

Let make an illustration of gradient descent: for $\theta \in R^2$, consider quadratic function $f(\theta) = \frac{1}{2}\theta^TP\theta + q^T\theta$ for $P$ positive definite (all positive eigenvalues). Illustration of gradient descent with different step sizes:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250228101926028.png" alt="image-20250228101926028" style="zoom:33%;" />

We can see the oscillation over time for diffirent step size.

### Newton’s Method

We can see from gradient descent that step is critical to optimization. So how to find the best step size to make less oscillation? We can see the solution from Newton's Method. **Newton's Method** is one way to integrate more “global” structure into optimization methods, which <u>scales gradient according to inverse of the Hessian (matrix of second derivatives</u>.
$$
\theta_{t+1} = \theta_t - \alpha(\nabla^2_\theta(f(\theta_t)))^{-1} \nabla_\theta f(\theta_t)
$$
where $\nabla^2_\theta(f(\theta_t))$ is the Hessian, $n \times n$ matrix of all second derivatives.

It's equivalent to approximating the function as quadratic using second-order Taylor expansion, then solving for optimal solution.

Full step given by $\alpha = 1$, otherwise called a damped Newton method. Newton’s method (will $\alpha = 1$) will optimize quadratic functions in one step.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250228102909317.png" alt="image-20250228102909317" style="zoom:50%;" />

However, it's not of that much practical relevance to deep learning for two reasons:

- We **can’t efficiently solve for Newton step**, even using automatic differentiation (though there are tricks to approximately solve it)
- For **non-convex optimization**, it’s very unclear that we even want to use the Newton direction

### Momentum

Can we find “middle grounds” that are **as easy to compute as gradient descent**, but which **take into account more “global” structure** like Newton’s method.

One common strategy is to use *momentum* update, that takes into account <u>a moving average of multiple previous gradients</u>.
$$
u_{t+1} = \beta u_t + (1 - \beta) \nabla_\theta f(\theta_t)
$$

$$
\theta_{t+1} = \theta_{t} - \alpha u_{t+1}
$$

where $\alpha$ is step size as before, and $\beta$ is momentum averaging parameter.

Compare to gradient descent, momentum “smooths” out the descent steps, but can also introduce other forms of oscillation and non-descent behavior. It's frequently useful in training deep networks in practice.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250228223449546.png" alt="image-20250228223449546" style="zoom:50%;" />

> When $\beta = 0$, momentum method is equivalent to gradient descent method.

#### “Unbiasing” momentum terms

The momentum term $u_t$ (if initialized to zero, as is common), will be smaller in initial iterations than gradient descent method. To “unbias” the update to have equal expected magnitude across all iterations, we can use the update:
$$
\theta_{t+1} = \theta_t - \alpha \frac{u_{t+1}}{1 - \beta^{t+1}}
$$
<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250228224507102.png" alt="image-20250228224507102" style="zoom:50%;" />

#### Nesterov Momentum

One useful tricks in the notion of Nesterov momentum (or Nesterov acceleration), which **computes momentum update at “next” point**.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250228225031519.png" alt="image-20250228225031519" style="zoom:33%;" />

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250228225054333.png" alt="image-20250228225054333" style="zoom:50%;" />

It's “good” thing for convex optimization, and (sometimes) helps for deep networks

### Adam

The scale of the gradients can vary widely for different parameters, especially e.g. across different layers of a deep network, different layer types. So we need a adaptive gradient methods to estimate this scale over iterations and then re-scale the gradient update accordingly.

Most widely used adaptive gradient method for deep learning is Adam algorithm, which **combines momentum and adaptive scale estimation**:
$$
u_{t+1} = \beta_1 u_t + (1 - \beta_1) \nabla_\theta f(\theta_t)
$$

$$
v_{t+1} = \beta_2 u_t + (1 - \beta_2) (\nabla_\theta f(\theta_t))^2
$$

$$
\theta_{t+1} = \theta_t  - \alpha \frac{u_{t+1}}{v_{t+1}^{1/2} + \epsilon}
$$

There are alternative universes where endless other variants became the “standard” (no unbiasing? average of absolute magnitude rather than squared? Nesterov-like acceleration?) but Adam is well-tuned and hard to uniformly beat:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250228231145042.png" alt="image-20250228231145042" style="zoom:50%;" />

### Stochastic variants

All the previous examples considered batch update to the parameters, but the single most important optimization choice is to use **stochastic variants**. Recall our machine learning optimization problem:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250228231302532.png" alt="image-20250228231302532" style="zoom:33%;" />

which is the minimization of an empirical expectation over losses.

We can **get a noisy (but unbiased) estimate of gradient by computing gradient of the loss over just a subset of examples (called a minibatch)**

We can use gradient descent as an example: SGD(Stochastic Gradient Descent) which repeating gradient descent for batches $B \subset {1,...,m}$:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250228231726418.png" alt="image-20250228231726418" style="zoom:33%;" />

Instead of taking a few expensive, noise-free, steps, we take many cheap, noisy steps, which **ends having much strong performance per compute**:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250228231837273.png" alt="image-20250228231837273" style="zoom:50%;" />

## Initialization

Recall that we optimize parameters iteratively by stochastic gradient descent:
$$
W_i := W_i - \alpha \nabla_{W_i} l(h_\theta(X), y)
$$
But how do we choose the initial values of $W_i$, $b_i$? We may initialize them to 0 which is not a good idea. Recall the manual backpropagation forward/backward passes (without bias):

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250228232345276.png" alt="image-20250228232345276" style="zoom:33%;" />

### Key idea #1: Choice of initialization matters

Let’s just initialize weights “randomly”, e.g., $W_i ∼ N(0, \delta^2 I)$. The choice of variance $\delta^2 $ will affect two (related) quantities:

- The norm of the forward activations $Z_i$
- The norm of the the gradients $ \nabla_{W_i}l(h_\theta(X), y)$

We make a illustration on MNIST with n = 100 hidden units, depth 50, ReLU nonlinearities :

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250301163457625.png" alt="image-20250301163457625" style="zoom:50%;" />

### Key idea #2: Weights don’t move “that much”

You might have the picture in your mind that the parameters of a network converge to some similar region of points regardless of their initialization.

This is not true. Weights often stay much closer to their initialization than to the “final” point after optimization from different. So initialization really matters.

### Find Good Initialization Way

For **linear activation**, we consider independent random variables: $ x \in N(0, 1), w \in N(0, \frac{1}{n}I)$. Then 
$$
E(x_i w_i) = E(x_i)E{w_i} = 0, Var(x_i w_i)=Var(x_i)Var(w_i) = 1/n
$$
So $E(w^Tx) = 0, Var(w^Tx) = 1$.Thus, informally speaking if we used a linear activation and $z_i \in N(0, 1), w_i \in N(0, \frac{1}{n}I)$, then $z_{i+1} \in N(0, 1)$. When mantains the same property.

If we use a **ReLU nonlinearity**, then “half” the components of $z_i$ will be set to zero, so we need twice the variance on $W_i$ to achieve the same final variance, hence $W_i \in N(0, \frac{2}{n}I)$. It's called **Kaiming normal initialization**.

## Tricks to Nerual Network Library

In this part, we'll introduce some tricks to train a good nerual network.

### Normalilzation

Initialization matters a lot for training, and can vary over the course of training to no longer be “consistent” across layers / networks But remember that a “layer” in deep networks can be any computation at all. We just add layers that “fix” the normalization of the activations to be whatever we want!

#### Layer normalization

 let’s normalize (mean zero and variance one) activations at each layer; this is known as **layer normalization**.
$$
\hat{z_{i+1}} = \sigma_i(W^Tz_i + b_i)
$$

$$
z_{i + 1} = \frac{\hat{z_{i+1}} - E(\hat{z_{i+1}})}{\sqrt{Var(\hat{z_{i+1}}) + \epsilon}}
$$

> Also common to add an additional scalar weight and bias to each term (only changes representation e.g., if we put normalization prior to nonlinearity instead)

Then $z_{i+1} \in N(0, 1)$, which mantains the property. We can known from the illusration of layer normalization that it “fixes” the problem of varying norms of layer activations (obviously):

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250301165249446.png" alt="image-20250301165249446" style="zoom:50%;" />

However, there is a side effect, mean/var of a layer manbe a good feature some tasks. So in practice, for standard FCN, harder to train resulting networks to low loss (relative norms of examples are a useful discriminative feature)

#### Batch normalization

Let’s consider the matrix form of our updates:
$$
\hat{Z_{i+1}} = \sigma(Z_iW_i +b^T)
$$
Then layer normalization is equivalent to <u>normalizing the rows of this matrix</u>

What if, instead, we normalize it’s columns? This is called **batch normalization**, as we are normalizing the activations over the minibatch.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250301170836543.png" alt="image-20250301170836543" style="zoom:50%;" />

There is dependency problem in inference time. It makes the predictions for each example dependent on the entire batch. However, we cannot get the future prediction in inference time.

Common solution is to compute a running average of mean/variance for all features at each layer $\hat{\mu_{i+1}},\hat{\sigma_{i+1}}^2$, and at test time normalize by these quantities.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250301172600475.png" alt="image-20250301172600475" style="zoom:33%;" />

### Regularization

**Regularization** is the process of “limiting the complexity of the function class” in order to ensure that <u>networks will generalize better to new data</u>; typically occurs in two ways in deep learning.

- *Implicit regularization* refers to the manner in which our existing algorithms (namely SGD) or architectures already limit functions considered.
  - E.g.: we aren’t actually optimizing over **“all neural networks”**, we are optimizing over all neural networks considered by SGD, with a given weight initialization.
- *Explicit regularization* refers to modifications made to the network and training procedure explicitly intended to regularize the network

#### ℓ2 Regularization(weight decay)

Classically, **the magnitude of a model’s parameters** are often a reasonable proximation for complexity, so we can minimize loss while also keeping parameters small.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250301182410598.png" alt="image-20250301182410598" style="zoom:33%;" />

Results in the gradient descent updates:
$$
W_i := W_i - \alpha \nabla_{W_i}l(h(X), y) - \alpha \lambda W_i = (1 - \alpha \lambda)W_i - \alpha \nabla_{W_i}l(h(X), y)
$$


I.e., at each iteration we shrink the weights by a factor $(1 - \alpha \lambda)$ before taking the gradient step.

ℓ2 regularization is exceedingly common deep learning, often just rolled into the optimization procedure as a “weight decay” term.

However, recall our optimized networks with different initializations:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250301183056354.png" alt="image-20250301183056354" style="zoom:33%;" />

Parameter magnitude may be a bad proxy for complexity in deep networks.

#### Dropout

Another common regularization strategy: randomly set some fraction of the activations at each layer to zero.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250301183416185.png" alt="image-20250301183416185" style="zoom:33%;" />

Not unlike BatchNorm, it seems very odd on first glance: doesn’t this massively change the function being approximated?

Dropout is frequently cast as making networks “robust” to missing activations. We can be instructive to consider Dropout as **bringing a similar stochastic approximation as SGD to the setting of individual activations**

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250301183642884.png" alt="image-20250301183642884" style="zoom:33%;" />



## Implementation



