---
title: 'DLSys: Transformers and Attention'
date: 2025-04-01 09:52:23
tags:
- transformer
categories:
- dlsys

pin: false
math: true
---

## Theory

### The two approaches to time series modeling

Let’s recall our basic time series prediction task from the previous posts. More fundamentally, a time series prediction task is the task of predicting:
$$
y_{1:T}=f_\theta(x_{1:T})
$$
<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240828130234284.png" alt="image-20240828130234284" style="zoom:50%;" />

where $y_t$ can depend only on $x_{1:t}$. There are mainly two approach to do so: **latent state approach** and **direct prediction**.

#### The RNN “latent state” approach

We have already seen the RNN approach to time series: **maintain “latent state” $h_t$ that summarizes all information up until that point**.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240828130718884-20240828130724791.png" alt="image-20240828130718884" style="zoom: 33%;" />

- **Pros**: Potentially “infinite” history, compact representation. It means the time series length is not limited and has low memory to store the previous context.
- **Cons**:  Long “compute path” between history and current time ⟹ vanishing / exploding gradients, hard to learn. A single state $h_t$ is hard to represent long context, easily lost the previous context.

#### The “direct prediction” approach

To avoid vanishing/exploding gradients(lose context/context is hard to store), we can also directly predict output $y_t$:
$$
y_t = f_\theta(x_{1:t})
$$
$f_\theta$ must be a function that can make predictions of differently-sized inputs.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240828131152642.png" alt="image-20240828131152642" style="zoom:33%;" />

- **Pros**:  Often can map from past to current state with shorter compute path. With a proper function $f_\theta$ , we can avoid vanishing/exploding gradients.
- **Cons**: No compact state representation, finite history in practice.

One of the most straightforward ways to specify the function $f_\theta$: (fully) convolutional networks, a.k.a. temporal convolutional networks (TCNs). The main constraint is that **the convolutions be causal**: $z^{i+1}_t$ can only depend on $z^{i}_{t-k:t}$.

Many successful applications: e.g. WaveNet for speech generation ([van den Oord et al., 2016](https://deepmind.google/discover/blog/wavenet-a-generative-model-for-raw-audio/))

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240828132255771.png" alt="image-20240828132255771" style="zoom:67%;" />

Despite their simplicity, CNNs have a notable disadvantage for time series prediction: the receptive field of each convolution is usually relatively small ⟹ need deep networks to actually incorporate past information. For example, WaveNet can achive a receptive field of 16 using 4 layer. For very long sequence, it will need very deep network. There are several solutions:

- **Increase kernel size**: also increases the parameters of the network
- [**Pooling layers**](https://medium.com/@abhishekjainindore24/pooling-and-their-types-in-cnn-4a4b8a7a4611): not as well suited to dense prediction, where we want to predict all of $y_{1:T}$. We'll lose some predictions because pooling will decrease the size of input.(alose decrease the size of output)
- **Dilated convolutions**: “Skips over” some past state / inputs. But we'll lose some context which may be important.

As we can see, CNN is not well suited for time series prediction. We'll introduce a new arch of network: transformer which will overcome the cons of CNN.



### Self-attention and Transformers

#### Self Attention

Let's first talk about the important part of transformer: **Self Attention**! “Attention” in deep networks generally refers to **any mechanism where individual states are weighted and then combined**.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240829085656637.png" alt="image-20240829085656637" style="zoom:50%;" />

Attention is used originally in RNNs when one wanted to combine latent states over all times in a more general manner than “just” looking at the last state. Let's define general attention in math:
$$
z_t = \theta^Th_t^k
$$

$$
w = softmax(z)
$$

$$
\bar{h} = \sum_{t=1}^T(w_th_t^k)
$$

> The **softmax function** converts a vector of *K* real numbers into a [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution) of *K* possible outcomes. 

**Self-attention** refers to a particular form of attention mechanism. Given three inputs $K,Q,V \in R^{T \times d}$, (“queries”, “keys”, “values”, in one of the least-meaningful semantic designations we have in deep learning).

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240829090838397.png" alt="image-20240829090838397" />

we define the self attention operation as:
$$
SelfAttetion(Q,K,V)=softmax(\frac{QK^T}{d^{1/2}})V
$$
Where the input is $X \in R^{T \times n}, W_k \in W^{n \times d}, W_Q \in W^{n \times d}, W_V \in W^{n \times d}$, we can simple calculate $Q, K, V$ as follows:
$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$



Compare to the attention used in rnn, we use input to replace the hidden state $h$.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240829092012276.png" alt="image-20240829092012276" style="zoom:50%;" />

Properties of self-attention:

- Invariant (really, equivariant) to permutations of the $Q, K, V$matrices 
- Allows influence between $q_t,k_t, v_t$over all times without increase parameter size.(compare to CNN, in order to increase reception field, we need to increase kernel size --> increase parameter size)
-  Compute cost is $O(T^2 + 2Td)$ (cannot be easily reduced due to nonlinearity applied to full $T \times T$ matrix)
  - softmax const $T^2$
  - two matrix multiplication cost $Td$

#### Transformer

A simple transoformer block consist of  self-attention mechanism and other network blocks.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240829092949134.png" alt="image-20240829092949134" style="zoom:50%;" />

In more detail, the Transformer block has the following form:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240829093117580.png" alt="image-20240829093117580" style="zoom: 33%;" />

The Transformer architecture uses a series of attention mechanisms (and feedfoward layers) to process a time series:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240829093229868.png" alt="image-20240829093229868" style="zoom: 50%;" />

Which can be form use math equation below:
$$
Z^{(i+1)}=TransformerBlock(Z^{(i)})
$$
All time steps (in practice, within a given time slice) are **processed in parallel**, avoids the need for sequential processing as in RNNs.

We can apply the Transformer block to the “direct” prediction method for time series, instead of using a convolutional block.

- Pros:
  - Full receptive field within a single layer (i.e., can immediately use past data) 
  - Mixing over time doesn’t increase parameter count (unlike convolutions)

- Cons:
  - All outputs depend on all inputs (no good e.g., for autoregressive tasks)  -- the latent cortex is more important.
  - No ordering of data (remember that transformers are equivariant to permutations of the sequence)--the position masters.

To solve the cons of transformer, we introduce two techniques: **masked self-attention**, and **Positional encodings**.

#### Masked self-attention

To solve the problem of “acausal” dependencies, we can mask the softmax operator to assign zero weight to any “future” time steps.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240829094005050.png" alt="image-20240829094005050" style="zoom: 50%;" />

Note that even though technically this means we can “avoid” creating those entries in the attention matrix to being with, in practice it’s often faster to just form them then mask them out.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240829094241280.png" alt="image-20240829094241280" style="zoom:50%;" />

#### Positional encodings

To solve the problem of “order invariance”, we can add a positional encoding to the input, which **associates each input with its position in the sequence**.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240829094534849.png" alt="image-20240829094534849" style="zoom:50%;" />

and where $w_i, i = 1,...,n$ is typically chosen according to a logarithmic schedule. Really, add positional encoding to d-dimensional projection of $T$

### Transformers beyond time series

Recent work has observed that transformer blocks are extremely powerful beyond just time series

- Vision Transformers: Apply transformer to image (represented by a collection of patch embeddings), works better than CNNs for large data sets
- Graph Transformers: Capture graph structure in the attention matrix 

In all cases, some challenges are: 

- How to represent data such that $O(T^2)$ operations are feasible 
- How to form positional embeddings 
- How to form the mask matrix

## Implementation

The runnable colab implementation is [here](https://colab.research.google.com/drive/1C7aKGLY97MxoObOXV0PfE0kmIy0-YIsw?usp=sharing)
