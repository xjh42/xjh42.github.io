---
title: 'DLSys: Convolutional Networks'
date: 2024-03-20 09:38:36
tags:
- transformer
categories:
- dlsys

pin: false
math: true
---

In this post, we'll introduce Convolutional Networks(CNN).

# Theory

## Convolution Operator

So far we only consider fully connected networks, which treat input images as vectors(size is n ), and use a large weight matrix( $W \in R^{n \times d}$ ) to map input vector to a feature vector. 

- **Parameter will explode**: This creates a substantial problem as we attempt to handle larger images: a 256x256 RGB image ⟹ ~200K dimensional input ⟹ mapping to 1000 dimensional hidden vector requires 200M parameters (for a single layer)

- **The structure feature of image will lose** : Another problem is this operation **does not capture any of the “intuitive” invariances that we expect to have in images** (e.g., shifting image one pixel leads to very different next layer). It means we use full image pixels to predict single value in next layer.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240830095205371.png" alt="image-20240830095205371" style="zoom:67%;" />

To solve this, we can use CNN, first let's see the convolution operator.

###  Convolutions can “Simplify” Deep Networks

Convolutions combine two ideas that are well-suited to processing images 

- Require that **activations between layers occur only in a “local” manner**, and treat hidden layers themselves as spatial images --> preserve image structure features 
-  **Share weights** across all spatial locations --> reduce parameters size.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240830095715641.png" alt="image-20240830095715641" style="zoom:50%;" />

Compare to full connected network, convolution can:

- Drastically reduces the parameter count.(256x256 grayscale image ⟹ 256x256 single-channel hidden layer: 4 billion parameters in fully connected network to 9 parameters in 3x3 convolution)
- Captures (some) “natural” invariances (Shifting input image one pixel to the right shifts creates a hidden shifts the hidden unit “image”)

Let's see how convolution works in details

### Convolutions in detail

Convolutions are a basic primitive in many computer vision and image processing algorithms. Convolution operator is to **“slide” the weights $k \times k$ weight $w$  (called a filter, with kernel size $k$) over the image to produce a new image, written $y = z * w$**.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240830100303185.png" alt="image-20240830100303185" style="zoom: 33%;" />

let's see how to compute $y_{11}$:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240830124621705.png" alt="image-20240830124621705" style="zoom: 33%;" />

The rest value of $y$ is calculated similarly.

### Convolutions in Image Processing

Convolutions (typically with prespecified filters) are a common operation in many computer vision applications: convolution networks just move to learned filters.

For conditional image programming, we use predefined filter, like Gasssian Filter, Image gradient Filter.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240830124932921.png" alt="image-20240830124932921" style="zoom: 33%;" />

### Convolutions in deep networks

Convolutions in deep networks are virtually always multi-channel convolutions: **map multi-channel (e.g., RGB) inputs to multi-channel hidden units**. Multi-channel convolutions contain a convolutional filter for each input-output channel pair, single output channel is sum of convolutions over all input channels. It shows below.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240830125227168.png" alt="image-20240830125227168" style="zoom: 50%;" />

Let's see how to define it in math:

- $x \in R^{h \times w \times c_{in}}$ denotes $c_{in}$ channel, size $h \times w$ image input
- $z \in R^{h \times w \times c_{out}}$ denotes $c_{out}$ channel, size $h \times w$ image out
- $W \in R^{c_{in} \times c_{out} \times k \times k}$ (order 4 tensor) denotes convolutional filter

$$
z[:,:,s] = \sum_{r=1}^{c_{in}}{x[:,:,r] * W[r,s,:,:]}
$$



The math equation is hard to understand. There is, in my view, a more intuitive way to think about multi-channel convolutions: **they are a generalization of traditional convolutions with scalar multiplications replaced by matrix-vector products.**

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240830130039529.png" alt="image-20240830130039529" style="zoom:80%;" />

## Elements of Practical Convolutions

Naive convolution is hard to fit different condition. So there are serveral techniques to make convolution more practical.

### Padding--Preserve resolution 

“Naïve” convolutions produce a smaller output than input image. If we want to get a same resolution image, we need use padding technique.Be careful, padding is only work for **odd kernel size**.

For (odd) kernel size $k$,  pad input with $(k-1)/2$ zeros on all sides, results in an output that is the same size as the input

- There are serval variants like **circular padding**, **padding with mean values**, etc

  <img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240830130634789.png" alt="image-20240830130634789" style="zoom:80%;" />

### Strided Convolutions / Pooling--decrese resolution 

Given input matrix and filter, convolution output a fixed-size of output matrix,  don’t naively allow for representations at different “resolutions”. **If you want to a self-defined size of output matrix, you can use either stride convolution or pooling techniques**.

- **Pooling** : incorporate max or average pooling layers to aggregate information

  <img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240830131034673.png" alt="image-20240830131034673" style="zoom:50%;" />

- **Strided Convolutions:** slide convolutional filter over image in increments >1 (= stride)

  <img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240830131143353.png" alt="image-20240830131143353" style="zoom:50%;" />

  ### Grouped Convolutions

   For large numbers of input/output channels, filters can still have **a large number of weights**, can lead to **overfitting + slow computation**.

  To solve this,  we can group together channels, so that **groups of channels in output only depend on corresponding groups of channels in input **(equivalently, enforce filter weight matrices to be block-diagonal)

  <img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240830131419161.png" alt="image-20240830131419161" style="zoom:50%;" />

  Given a simple example below, we can reduce filter parameter size from $R^{3 \times 3 \times k \times k}$ to $R^{ k \times k}$.

  <img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240830131952146.png" alt="image-20240830131952146" style="zoom: 50%;" />

  ### Dilations

   Convolutions each have a **relatively small receptive field size**. We'll lose the global context of the image.We can use dilation to solve this problem. 

  Dilate (spread out) convolution filter, so that it covers more of the image; note that getting an image of the same size again requires adding more padding.

  <img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240830132217024.png" alt="image-20240830132217024" style="zoom:80%;" />

## Differentiating Convolutions--convolution as single node in CG

Recall the convolution math definition:
$$
z[:,:,s] = \sum_{r=1}^{c_{in}}{x[:,:,r] * W[r,s,:,:]}
$$


If we use the basic operations like multiply/add/sum to form convolution and it's gradient. The computation graph will be large and cost a lot of memory. So we want to define convolution as basic operator. So it'll be only single node in a computation graph.

Recall that in order to integrate any operation into a deep network, we need to be able to multiply by its partial derivatives (adjoint operation). So if we define our operation:
$$
z = conv(x, W)
$$


how do we multiply by the adjoints:
$$
\bar{v}\frac{\partial conv(x, W)}{\partial x},\bar{v}\frac{\partial conv(x, W)}{\partial W}
$$
Let’s consider the simpler case of a matrix-vector product operation:
$$
z =Wx
$$


Then $\frac{\partial z}{\partial x} = W$, , so we need to compute the adjoint product:
$$
\bar{v}^TW \iff W^T\bar{v}
$$


In other words, for a matrix vector multiply operation $Wx$, computing the backwards pass requires multiplying by the transpose $W^T$. 

In the next section, we can convert convolution to matmul operation, then we can simply get the result of differentiation of convolution.

### Convolutions as matrix multiplication: Version 1

consider a 1D convolution to keep things a bit simpler:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240831123813936.png" alt="image-20240831123813936" style="zoom: 33%;" />

We can write a 1D convolution $x * w$ (e.g., with zero padding) as a matrix multiplication $\hat{W}x$  for some $\hat{W}$ properly defined in terms of the filter $w$.
$$
\begin{bmatrix} z_1 \\ z_2 \\ z_3 \\ z_4 \\ z_5 \end{bmatrix} = \begin{bmatrix} w_2 & w_3 & 0  & 0 & 0 \\ w_1 & w_2 & w_3  & 0 & 0 \\ 0 & w_1 & w_2  & w_3 & 0 \\ 0 & 0 & w_1  & w_2 & w_3 \\ 0 & 0 & 0  & w_1 & w_2 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \\ x_5 \end{bmatrix}
$$
By converting convolution to matmul, we can easily compute the gradient of convolution. Just compute the transponse of $\hat{W}$:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240831215657389.png" alt="image-20240831215657389" style="zoom: 40%;" />

Notice that the operation $\hat{W}^Tv$ it itself just a convolution with the “flipped” filter: $[w_3 \space w_2 \space w_1]$ ==> **adjoint operator $\bar{v}\frac{\partial conv(x, W)}{\partial x}$ just requires convolving $\bar{v}$ with a the flipped $W$**.



### Convolutions as matrix multiplication: Version 2

What about the other adjoint,$\bar{v}\frac{\partial conv(x, W)}{\partial W}$?

For this term, observe that we can also write the convolution as **a matrix-vector product treating the filter as the vector**.
$$
\begin{bmatrix} z_1 \\ z_2 \\ z_3 \\ z_4 \\ z_5 \end{bmatrix} = \begin{bmatrix} 0 & x_1 & x_2 \\ x_1 & x_2 & x_3 \\ x_2 & x_3 & x_4 \\ x_3 & x_4 & x_5 \\ x_4 & x_5 & 0 \end{bmatrix} \begin{bmatrix} w_1 \\ w_2 \\ w_3 \end{bmatrix}
$$


So adjoint requires multiplying by the transpose of this $x$-based matrix:
$$
\begin{bmatrix} 0 & x_1 & x_2 & x_3 & x_4 \\ x_1 & x_2 & x_3 & x_4 & x_5 \\ x_2 & x_3 & x_4 & x_5 & 0 \end{bmatrix}
$$


# Implementation

The reference implementation is [here](https://colab.research.google.com/drive/1DCQHM-rEI04YS8OlktWp0S5MuwYb74Ok?usp=sharing)
