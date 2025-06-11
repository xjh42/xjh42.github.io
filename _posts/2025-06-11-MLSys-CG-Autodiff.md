---
title: 'MLSys 1: Computation Graph and Autodiff'
date: 2025-04-31 09:52:23
tags:
- autodiff
categories:
- mlsys
pin: false
math: true
---


In this post, we'll introduce the core components of ML system: **computation graph** and **automatic differentiation**. We'll first introduce the background which is dl computation. Then we'll introduce the deep learning workload. Then we'll introduce the computation graph to represent the computation. After that,  we'll introduce the automatic differentiation to solve the dl problem as a learning problem. Finally, we'll see how to design a deep learning framework using tensorflow and pytorch as examples.

## Background: DL Computation

When we are building systems, we need to first understand our workload, which primarily consists of machine learning and deep learning computations. Deep learning involves stacking many neural network layers to compose a large and effective model. For example, in image classification, we perform the following steps:

- Define the layers of the neural network
- Perform specific computations on each layer
- Forward the images through the network
- Obtain the prediction for the given image

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250603092119184.png" alt="image-20250603092119184" style="zoom:30%;" />

A neural network will not work unless it is trained; to train it, first predict with forward progagation, compute the gradients of the loss, and finally perform backward propagation to update the parameters based on the computed gradients. The equation is:
$$
\theta^{(t+1)} = f(\theta^{(t)}, \nabla_L(\theta^{(t)}, D^{(t)}))
$$


- $\theta^{(t)}$:  Parameters of the model at iteration $t$
- $L(\theta, D)$: Loss function, which depends on the problem, e.g., L2 loss, softmax loss
- $\nabla_L(\theta^{(t)}, D^{(t)})$: : Gradient of the loss function $L$ with respect to the parameters $\theta^{(t)}$  given the training data $D^{(t)}$
- $D^{(t)}$: Training data (e.g., images, text, video, audio)
- $T$: Total number of iterations until convergence

In deep learning, the objective is to optimize the set of parameters $\theta$ that will minimize the loss, which is achieved by continuously feeding data into the neural network, calculating gradients, and applying parameter updates until the parameters converges with gradients as close to 0 as possible. The optimization method is a procedure used to find a set of parameters that minimize the loss. Common optimization algorithms include SGD, Adam, Newton, etc.

There are three important components:

- **Data**: it contains images, text, audio, table, etc.
- **Model**: CNNs, RNNs, Transformers, MoEs, etc.
- **Compute**: CPUs,GPUs/TPUs/LPUs, M1/M2/M3/M4, FPGA, etc.

##  Understand our Workloads: Deep Learning

In system building, it is often impractical to support all possible models. Instead, we focus on identifying the most important workloads that should solve about 80% of the problem. System building is the process of uncovering the most critical factors that define these workloads.

- **Most Important Models**: CNNs (Convolutional Neural Networks), RNNs (Recurrent Neural Networks), Transformers, MoEs (Mixture-of-Experts).
- **Most Important Algorithms**: SGD (Stochastic Gradient Descent) and its variants, such as Adam.

The key to system building is identifying the most important X in Y (X ⊂ Y ) to define the system’s abstractions, such as X = ResNet and Y = CNNs.

### CNN

**CNNs** is an important network architecture which has many applications. 

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250603094019056.png" alt="image-20250603094019056" style="zoom:30%;" />

The key components of CNN is to **convolve the filter with the image**. In details, it use a filter(a smaller matrix) to slide over the image spatially and compute dot products.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250603094242204.png" alt="image-20250603094242204" style="zoom:33%;" />

Why CNN is powerful in computer vision area?  From  [[zeiler and fergus 2013](https://arxiv.org/pdf/1311.2901)] We know that we can stack conv layers to get image low level features to high level features.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250603131829926.png" alt="image-20250603131829926" style="zoom:30%;" />

The **top 3 model** in CNN is:

- [[AlexNet](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)]: use multiple conv layers to do image classification.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250603132254549.png" alt="image-20250603132254549" style="zoom:33%;" />

- [[ResNet](https://arxiv.org/pdf/1512.03385)]: first introduce the residual module to ease the training of networks that are substantially deeper than those used previously.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250603132431596.png" alt="image-20250603132431596" style="zoom:33%;" />

- [[U-Net](https://arxiv.org/abs/1505.04597)]: used from image segmentation.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250603132527552.png" alt="image-20250603132527552" style="zoom:33%;" />

Finally, what's the most important component in CNNs?

- **Conv**: Conv1d, Conv2d, conv3d, esp. 3x3-conv2d.
- **Matmul (linear)**: $C = A \times B$
- **Softmax**: $\sigma = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$

- **Elementwise operations:** ReLU, add, sub, Pooling, normalization, etc

### RNN

RNNs possess a unique ability to model both one-to-many and many-to-one relationships, such as mapping a sequence of inputs to a single output label. They use sequential data and this data can be dynamic, allowing for an arbitrary number of inputs and outputs. 

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250603133214131.png" alt="image-20250603133214131" style="zoom:33%;" />

The key idea for RNN is that RNNs use an internal state that is updated as a sequence is processed.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250603133416202.png" alt="image-20250603133416202" style="zoom:33%;" />

Theoretically, any neural network could be made into a RNN. The following figure shows how the building blocks of the model could be embedded with by any kind of neural network, such as a CNN or MLP.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250603133546870.png" alt="image-20250603133546870" style="zoom:33%;" />

The top 3 model in RNNs:

- [Bidirectional RNNs](https://ieeexplore.ieee.org/document/650093): Makes computations in both directions, left-to-right and right-to-left.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250604092355825.png" alt="image-20250604092355825" style="zoom:50%;" />

- [LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf): Has been widely adopted in time-series analysis due to its ability to remember or forget information as required.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250604092505781.png" alt="image-20250604092505781" style="zoom: 50%;" />

- [GRU](https://arxiv.org/abs/1412.3555): A powerful, simplified version of LSTM

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250604092636745.png" alt="image-20250604092636745" style="zoom:50%;" />

The most important component inRNNs:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250604092831695.png" alt="image-20250604092831695" style="zoom:50%;" />

- **Matmul**
- **Elementwise nonlinear**: ReLU, Tanh, sigmoid, etc

Finally, why ChatGPT was not built using RNNs?

- Problem 1: **forgetting** (h * 0.9 * 0.9 * … -> 0)
- Problem 2: **lack of parallelizability.** Both forward and backward passes have **O(sequence length) unparallelizable operators**. i.e., a state cannot be computed before all previous states have been computed Inhibits training on very long sequence 

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250604093214835.png" alt="image-20250604093214835" style="zoom:47%;" />

We'll see how transformer(attention mechanism) to solve these issues.

### Transformer

Researchers at Google proposed the influential paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) to fix the problems in RNNs.

Let's see how attention to enable parallelism. The basic idea is that attention <u>treats each position’s representation as a query to access and incorporate information from a set of values</u>.  Since each position operates independently, there are no sequential dependencies, allowing for parallel processing.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250604093503083.png" alt="image-20250604093503083" style="zoom:50%;" />

In the following figure, each hidden position in a given layer is computed by treating its current representation as a query, which then attends to the keys and values from all positions in the previous layer. In this way, all the positions with label ”1” could be computed in parallel from layer 0. Because of its sequential independency, the attention mechanism is perfect for GPUs to parallelize. In other words, Attention is massively parallelizable:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250604093816441.png" alt="image-20250604093816441" style="zoom:50%;" />

With the invention of attention mechanisms, we can construct the transformer model. As shown in following figure, the inputs go through attentions, LayerNorms - normalization operation element-wise, MLPs, and finally output some value.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250604094056754.png" alt="image-20250604094056754" style="zoom:50%;" />

The transformer model has two types: 

- **Encoder**: The most famous example of an encoder transformer is BERT. 
-  **Decoder**: The most famous example of a decoder transformer is GPT.

The top-3 models in transformer are:

- [[Bert](https://arxiv.org/abs/1810.04805)]:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250605082717345.png" alt="image-20250605082717345" style="zoom:50%;" />

- [[GPT/LLMs](https://arxiv.org/abs/2005.14165)]

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250605082850341.png" alt="image-20250605082850341" style="zoom:40%;" />

- [[DiT: diffusion](https://arxiv.org/abs/2212.09748)]

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250605082955414.png" alt="image-20250605082955414" style="zoom:53%;" />

Most Important Components in Transformers:

- **Transformers** = Attention + MLP + something else
- **Attention**: Matmul; Softmax; Normalization 
- **MLP**：Matmul
- Miscs: **Layernorm**, **GeLU**, etc.

### MoE

MoE means Mix of Experts which use multi-MLP as experts and use router function to choose MLP. The basic idea is voting from many experts could be better than one expert due to sparsity.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250605083458987.png" alt="image-20250605083458987" style="zoom:50%;" />

Latest LLMs like Grok, Mixtral, GPT4, Deepseek-v3 are mostly MoEs. The Novel components in MoE are router. It's a choose top-k function.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530130226400.png" alt="router" style="zoom:40%;" />

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530132240875.png" alt="router" style="zoom:33%;" />

The router mainly contains: matmul, softmax. It's hard to train router due to the instability.

## Computation Graph

The components in models like softmax/matmul are math primitives. We want to express as many as model as possible using one set of programming interface by connecting math primitives. In this section, we'll introduce computation graph to express models by connecting math primitives. 

The computation graph contains nodes and edges.

- **Node**:  1). represents the computation (operator); 2). represents the output tensor of the operator; 3). represents an input constant tensor if it is not a compute operator.
- **Edge**:  represents the data dependency (data flowing direction)

The following figure shows the computation graph for:
$$
x_1 = 3; x_2 = 0.5; f = x_1 + e^{1.5 \times x_1 + 2.0 \times x_2}
$$


<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250605090015310.png" alt="image-20250605090015310" style="zoom: 33%;" />

So how to do computation with computation graph?

1. Topological sorting of all nodes
2. Calculate the value for each node given its input

The  algorithm is showed as follows:

```
- Put all nodes into unprocessed queue
- Repeatedly, find a node without incoming edges from un-processed nodes
  - evaluate its value based on operation
  - remove the node from the queue and add it to processed queue
```

Most  systems, including **Pytorch/Autograd**, explicitly construct the computation graph. **TensorFlow** provide mini-languages for building computation graphs directly.

To implement computation graph, we just need to wrap all the things to a `Node` class. Let's use [autodidact](https://github.com/mattjj/autodidact) as example:

- `Node` class, with attributes
  - `value`: the actual value computed on a particular set of inputs 
  - `fun`: the primitive operation defining the node 
  - `args` and `kwargs`: the arguments the op was called with
  - `parents`: the parent Nodes which mimic the edge

To connect math primitives, [autodidact](https://github.com/mattjj/autodidact)'s NumPy module provides primitive ops which look and feel like NumPy functions, but secretly build the computation graph.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250605091156077.png" alt="image-20250605091156077" style="zoom:67%;" />



## Automatic Differentiation

A neural network will not work unless it is trained. In deep learning, we can treat the training process to a learning problem: <u>given a training set of input-output pairs $D = {(x_n, y_n)}_{n=1}^{N}$($x_n, y_n$ may both be vectors), to find the model parameters such that the model produces the most accurate output for each training input or a close approximation of it.</u>

To solve it, let's first consider a generic function minimization problem, where x is unknown variable:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250605134202157.png" alt="image-20250605134202157" style="zoom:23%;" />

The iterative update algorithm is:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250605134242950.png" alt="image-20250605134242950" style="zoom:23%;" />

so that $f(x_{t+1}) << f(x_t)$. The question is how to find Δ

We know that approximate equation for $f(x_t + Δx)$ is $f(x_t) + Δx^T \nabla f|_{x_t}$. To make $Δx^T \nabla f|_{x_t}$ smallest, we need let $Δx^T$ to be the opposite direction of $\nabla f|_{x_t}$. i.e. $Δx^T = -\nabla f|_{x_t}$

So the update rule is: 

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250605135408478.png" alt="image-20250605135408478" style="zoom:23%;" />

𝜂 is a hyper-parameter to control the learning rate.

So the core problem is how to **compute the gradient** for every parameter in an “arbitrary network”. 

### Gradient Calculation

#### Gradient Calculation by Definition 

We can use the definition to calculate gradient:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250605135800469.png" alt="image-20250605135800469" style="zoom:23%;" />

However, it's unpractical in reality. There are some problems:

- **slow**: evaluate $f$ twice to get one gradient
- **Error**: approximal and floating point has errors

But we can use finite differences to check our gradient calculations.

#### Gradient Calculation by Symbolic Differntiation

Instead, we can use **symbolic differentiation** which write down the formula and derive the gradient following partial derivative rules:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250606002505216.png" alt="image-20250606002505216" style="zoom:23%;" />

In modern framework like PyTorch or Tensorflow, they use **automatic differentiation** by using symbolic differentiation to calculate gradient.

#### Partial derivatives for Vectors

For ML, the variables are vectors/matrixs. So we need to know how to calculate partial derivatives for vectors. First, we introduce the core idea: **Jacobian Matrix**: given $y = f(x)$, y and x are vectors

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250606131841991.png" alt="image-20250606131841991" style="zoom:23%;" />

- To get rows of Jacobian Matrix: : keep y index, iter x index
- To get columns of Jacobian Matrix: keep x index, iter y index

We can do chain rules by using **Jacobian Matrix** to calculate partial derivatives for vectors:
$$
\bar{x} = J \bar{y}
$$
For example, given $y=Wx$, the derivatives for x is: $\bar{x} = W^T\bar{y}$

So how to implement **Vector-Jacobian Product (VJP)**? For each primitive operation, we must specify VJPs for each of its arguments.

Let's see how [autodidact](https://github.com/mattjj/autodidact) implement VJP. It uses `defvjp` (defined in `core.py`)  which is a convenience routine for registering VJPs. 

```python
def defvjp(fun, *vjps, **kwargs):
    """Register vector-Jacobian product functions.

    Let fun(x, y, ...) = ans be a function. We wish to register a
    vector-Jacobian product for each of fun's arguments. That is, functions

      vjp_x(g, ans, x, y, ...) = g df/dx
      vjp_y(g, ans, x, y, ...) = g df/dy
      ...

    This function registers said callbacks.

    Args:
      fun: function for which one wants to define vjps for.
      *vjps: functions. vector-Jacobian products. One per argument to fun().
      **kwargs: additional keyword arugments. Only 'argnums' is used.
    """
    argnums = kwargs.get('argnums', count())
    for argnum, vjp in zip(argnums, vjps):
        primitive_vjps[fun][argnum] = vjp
```



### Autodiff Algorithm

There are two types of autodiff algorithm: **forward autodiff and backward autodiff**. We'll introduce them in details in following sub-sections.

#### Forward Autodiff

In forward autodiff, we define $\dot{v_i} = \frac{\partial v_i}{\partial x_1} $. We then compute each $\dot{v_i}$ following the forward order of the graph.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250606004058754.png" alt="image-20250606004058754" style="zoom: 30%;" />

In forward autodiff algorithm, we start from the input nodes and derive gradient all the way to the output nodes.For $f: R^n \rightarrow R^k$, we need 𝑛 forward passes to get the grad w.r.t. each input. However, **in ML: 𝑘 = 1 mostly, and 𝑛 is very large**. So we don't use this algorithm in practice.

#### Backward Autodiff

In forward autodiff, we define $\bar{v_i} = \frac{\partial y}{\partial v_i}$.  We then compute each $\bar{v_i}$ in the reverse topological order of the graph.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250606004957515.png" alt="image-20250606004957515" style="zoom:33%;" />

- Node with multiple outgoing edges:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250606005100516.png" alt="image-20250606005100516" style="zoom:43%;" />

How to derive the gradient of $v_1$?

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250606005141970.png" alt="image-20250606005141970" style="zoom:30%;" />

In general, For a $v_i$ used by multiple consumers:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250606005253100.png" alt="image-20250606005253100" style="zoom:33%;" />

The backward autodiff algorithm start from the output nodes and derive gradient all the way back to the input node. For $f: R^n \rightarrow R^k$ , we need 𝑘 backward passes to get the grad w.r.t. each input. In ML: 𝑘 = 1 and 𝑛 is very large which makes it's pratical in ML.

The implementation for backward autodiff algorithm is showed as follows:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250606005652067.png" alt="image-20250606005652067" style="zoom:40%;" />

A simple example for backward ad is showed as follows:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250606005808376.png" alt="image-20250606005808376" style="zoom:33%;" />

We can construct backward graph in a symbolic way (instead of concrete values). This graph can be reused by different input values.

## Deep Learning Framework Design

In this section, we'll take a look at deep learning framework in detail. We'll use Tensorflow and Pytorch as examples. All the materials are from their paper:

- TensorFlow: [[TensorFlow: A system for large-scale machine learning](https://arxiv.org/abs/1605.08695)]
- Pytorch: [[PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://arxiv.org/abs/1912.01703)]

A deep learning framework must have following features:

- **Expressive to specify any neural networks**: support future custom operators/layers
- **Productive for ML engineers**: it need to hide low-level details (no need to write cuda) and have automatic differentiation (no need to derive gradient calculation manually)
- **Efficient in large-scale training and inference**: it need to automatically scale to data and model size and do automatic hardware acceleration.

TensorFlow is an interface for expressing machine learning algorithms and an implementation for executing such algorithms. And PyTorch is a programming framework for tensor computation, deep learning, and auto differentiation. The comparison for different frameworks are showed in following table:

![image-20250606135621734](https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250606135621734.png)

### Tensorflow

#### Basics

The key idea of Tensorflow is to express a numeric computation as a computation graph. Graph nodes are **operations** with any number of inputs and outputs. Graph edges are *tensors*(multidimensional array) which flow between nodes.

In Tensorflow, A tensor is a multi-dimensional array. It's generalization to vector and matrix.

```python
tf.constant([[1, 2], [3, 4]]) # a 2x2 tensor with element type int32
tf.Tensor([[2 3] [4 5]], shape=(2, 2), dtype=int32) # we can specify the shape and dtype
```

For example, let's express the following equation $h = RELU(Wx + b)$ in Tensorflow:

```python
import tensorflow as tf
b = tf.Variable(tf.zeros((100,)))
W = tf.Variable(tf.random_uniform((784, 100), -1, 1))
x = tf.placeholder(tf.float32, (1, 784))
h = tf.nn.relu(tf.matmul(x, W) + b)
```

The computation graph is:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250610015616583.png" alt="image-20250610015616583" style="zoom:33%;" />

- Variable node: Variables are stateful nodes which output their current value. State is retained across multiple executions of a graph. often it's mostly parameters.

```python
tf.Variable(initial_value=None, trainable=None,name=None)
```

- Placeholder Node: Represent Inputs, Labels. It's value is fed in at execution time.

> No need to explicitly define Placeholder in Tensorflow v2

```python
x = tf.placeholder(tf.float32, (1, 784))
```

- Mathematical Operations: predefined functions in Tensorflow

```python
tf.linalg.matmul(a, b) # multiply two matrices
tf.math.add(a, b) # Add elementwise
tf.nn.relu(a) # Activate with elementwise rectified linear function
```

In TF v1, to deploy graph with a **session** which is a binding to a particular execution context (e.g. CPU, GPU).

```python
with tf.Session() as s:
 …
 s.run() 
```

To define loss, we use placeholder for labels, then build loss node using labels and prediction.

```python
prediction = tf.nn.softmax(...) #Output of neural network
label = tf.placeholder(tf.float32, [100, 10])
cross_entropy = -tf.reduce_sum(label * tf.log(prediction), axis=1)
```

Finally, we show how to do gradient computation:

```python
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

- `tf.train.GradientDescentOptimizer` is an `Optimizer` object
- `tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)` adds optimization operation to computation graph
- TensorFlow graph nodes have attached gradient operations
- Gradient with respect to parameters computed with Auto Differentiation

#### Design Principles

In tensorflow, all nodes return tensors. The node computation is different in v1/v2 version:

- In TF v1: it use metaprogramming which constructs the graph for the real computation and no computation occurs yet.
- TF v2 has eager mode, the computation is applied immediately (essentially constructing the graph and apply the computation)

Tensorflow use **deferred execution** which defines program i.e., symbolic dataflow graph w/ placeholders, essentially constructing the computation graph and  executes optimized version of program on set of available devices.

There are still problems. We need to support ML algos that contain conditional and iterative control flow like Recurrent Neural Networks (RNNs),  LSTMs and Autoregressive decoder. The solution is to add conditional (if statement) and iterative (while loop) programming constructs.

Let's take a look at the tensorflow architecture: 

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250611233107000.png" alt="image-20250611233107000" style="zoom:43%;" />

It's written in C++ and has different front ends for specifying/driving the computation.

The implementation of tensorflow is clear:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250611233331653.png" alt="image-20250611233331653" style="zoom:50%;" />

It calls GPU kernel per primitive operation and we can batch operations with custom C++. Like a compiled programming language, tensorflow enables basic type-safety within dataflow graph  which has compiled error at graph construction time.

#### Execution in Tensorflow

Similar to MapReduce, Apache Hadoop, Apache Spark, tensorflow use distributed execution workflow.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250611235108042.png" alt="image-20250611235108042" style="zoom:33%;" />

- Client construct the workfow and send to the master node:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250611235237679.png" alt="image-20250611235237679" style="zoom:33%;" />

- Master node use PS(parameter server) to do graph partition:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250611235412889.png" alt="image-20250611235412889" style="zoom:33%;" />

- Use worker to do actual computation.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250611235511808.png" alt="image-20250611235511808" style="zoom:33%;" />

Tensorflow has following assumptions:

- Fine grain operations: “It is unlikely that tasks will fail so often that individual operations need fault tolerance”
- Many learning algorithms do not require strong consistency

So in tensorflow, the solution of fault tolerance is to use user-level checkpointing. It has two operations:

- `save()`: writes one or more tensors to a checkpoint file
- `restore()`: reads one or more tensors from a checkpoint file

#### Mini-Tensorflow Implementation 

https://colab.research.google.com/drive/1bKkoIwctOH-vd39hAjNEDpYoU_YJna4N?usp=sharing
