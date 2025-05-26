---
title: 'Language Models 2: Pytorch'
date: 2025-04-01 18:52:23
tags:
- pytorch
categories:
- language models
pin: false
math: true
---

In order to build language model from scratch, we need know the tool that we'll use.  In this post, we'll use **pytorch**. So we'll go through the pytorch basics. The overview of the post is:

- We will discuss all the **primitives** needed to train a model.
- We will go bottom-up from tensors to models to optimizers to the training loop.
- We will pay close attention to efficiency (use of **resources**).

In particular, we will account for two types of resources: **Memory (GB)** and **Compute (FLOPs)**.

## Memory Accounting

### Tensor Basics

In pytorch, tensors are the basic building block for storing everything: parameters, gradients, optimizer state, data, activations. You can take  [[PyTorch docs on tensors\]](https://pytorch.org/docs/stable/tensors.html) as a more detailed reference. You can create tensors in multiple ways:

```python
x = torch.tensor([[1., 2, 3], [4, 5, 6]])  
x = torch.zeros(4, 8)  # 4x8 matrix of all zeros 
x = torch.ones(4, 8)  # 4x8 matrix of all ones 
x = torch.randn(4, 8)  # 4x8 matrix of iid Normal(0, 1) samples
```

You can also allocate but don't initialize the values:

```python
x = torch.empty(4, 8)  # 4x8 matrix of uninitialized values
```

You can use this to use some custom logic to set the values later. For example:

```python
nn.init.trunc_normal_(x, mean=0, std=1, a=-2, b=2)
```

### Tensor Memory

Almost everything (parameters, gradients, activations, optimizer states) in tensor are stored as **floating point numbers**. There four types of floating point number: `float32`, `float16`, `bfloat16`, `fp8`. We'll discuss them in details.

#### `float32`

The float32 data type (also known as fp32 or single precision) is the default type in pytorch. Traditionally, in scientific computing, float32 is the baseline; you could use double precision (float64) in some cases.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250525222807017.png" alt="image-20250525222807017" style="zoom:50%;" />

The value of float32 number can be represented as:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250525233101426.png" alt="image-20250525233101426" style="zoom:50%;" />



Let's examine memory usage of these tensors. Memory is determined by the (i) number of values and (ii) data type of each value.

```python
x = torch.zeros(4, 8)  # @inspect x
assert x.dtype == torch.float32  # Default type
assert x.numel() == 4 * 8
assert x.element_size() == 4  # Float is 4 bytes
assert get_memory_usage(x) == 4 * 8 * 4  # 128 bytes
```

`float32` cost a lot of memory. For example, one matrix in the feedforward layer of GPT-3:

```python
assert get_memory_usage(torch.empty(12288 * 4, 12288)) == 2304 * 1024 * 1024  # 2.3 GB
```

#### `float16`

The `float16` data type (also known as fp16 or half precision) cuts down the memory. The format of `float16` is showed as follows:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250525233217380.png" alt="image-20250525233217380" style="zoom:50%;" />

The value of float16 number can be represented as:
$$
value = (-1)^{sign} \times 2^{(E - 15)} \times (1 + \sum_{i=1}^{10}b_{10-i}2^{-i})
$$
You need to specify `dtype=torch.float16` to define `float16` tensor:

```python
x = torch.zeros(4, 8, dtype=torch.float16)  # @inspect x
assert x.element_size() == 2
```

However, the dynamic range (especially for small numbers) isn't great.

```python
x = torch.tensor([1e-8], dtype=torch.float16) 
assert x == 0  # Underflow!
```

If this happens when you train, you can get instability.

#### `bfloat16`

Google Brain developed `bfloat` (brain floating point) in 2018 to address this issue.`bfloat16` uses the same memory as float16 but has the same dynamic range as float32.The only catch is that the resolution is worse, but this matters less for deep learning. he format of `bfloat16` is showed as follows:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250526083943796.png" alt="image-20250526083943796" style="zoom:50%;" />

The value of `bfloat16` number can be represented as:
$$
value = (-1)^{sign} \times 2^{(E - 127)} \times (1 + \sum_{i=1}^{7}b_{7-i}2^{-i})
$$
We can see the advantage of `bfloat16` by showing following example:

```python
x = torch.tensor([1e-8], dtype=torch.bfloat16)  # @inspect x
assert x != 0  # No underflow!
```

Let's compare the dynamic ranges and memory usage of the different data types:

```python
float32_info = torch.finfo(torch.float32)  
float16_info = torch.finfo(torch.float16)  
bfloat16_info = torch.finfo(torch.bfloat16)
```

The result is:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250526084329655.png" alt="image-20250526084329655" style="zoom:50%;" />

#### fp8

In 2022, FP8 was standardized, motivated by machine learning workloads. It was developed by Nvidia: [Using FP8 with Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html).

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250526084543270.png" alt="image-20250526084543270" style="zoom:50%;" />

H100s support two variants of FP8: E4M3 (range [-448, 448]) and E5M2 ([-57344, 57344]). You can take the following paper as reference:  [[Micikevicius+ 2022\]](https://arxiv.org/pdf/2209.05433.pdf).

Different types of floating pointer number have different implications on training:

- Training with float32 works, but requires lots of memory.
- Training with fp8, float16 and even bfloat16 is risky, and you can get instability.
- use mixed precision training, see [mixed_precision_training](https://stanford-cs336.github.io/spring2025-lectures/?trace=var%2Ftraces%2Flecture_02.json&animate=1&step=112#)

## Compute Accounting

### Tensor on GPUs

By default, tensors are stored in CPU memory. You use following example to verify:

```python
x = torch.zeros(32, 32)
assert x.device == torch.device("cpu")
```

However, in order to take advantage of the massive parallelism of GPUs, we need to move them to GPU memory.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250526085205567.png" alt="image-20250526085205567" style="zoom:67%;" />

We can use following code to check GPU properties:

```python
if not torch.cuda.is_available():
    return
num_gpus = torch.cuda.device_count()  
for i in range(num_gpus):
    properties = torch.cuda.get_device_properties(i)
```

To transfer tensor from CPU to GPU, we can use following code:

```python
memory_allocated = torch.cuda.memory_allocated()  # @inspect memory_allocated
# Move the tensor to GPU memory (device 0)
y = x.to("cuda:0")
assert y.device == torch.device("cuda", 0)
```

Or we can directly create tensors on GPU:

```python
z = torch.zeros(32, 32, device="cuda:0")
new_memory_allocated = torch.cuda.memory_allocated()  
memory_used = new_memory_allocated - memory_allocated 
assert memory_used == 2 * (32 * 32 * 4)  # 2 32x32 matrices of 4-byte floats
```

### Tensor Operations

Most tensors are created from performing operations on other tensors. Each operation has some memory and compute consequence. 

#### Tensor Storage

What are tensors in PyTorch? PyTorch tensors are pointers into allocated memory with metadata describing how to get to any element of the tensor. To map tensor value to value in memory, we need to use `stride` variable in tensor metadata. You can take [[PyTorch docs\]](https://pytorch.org/docs/stable/generated/torch.Tensor.stride.html) as reference.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250526090130408.png" alt="image-20250526090130408" style="zoom:50%;" />

For example, we can define `x`:

```python
x = torch.tensor([
        [0., 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ])
```

To go to the next row (dim 0), skip 4 elements in storage.

```python
assert x.stride(0) == 4
```

To go to the next column (dim 1), skip 1 element in storage.

```python
assert x.stride(1) == 1
```

To find an element:

```python
r, c = 1, 2
index = r * x.stride(0) + c * x.stride(1) 
assert index == 6
```

#### Tensor Slicing

Many operations simply provide a different **view** of the tensor. This does not make a copy, and therefore mutations in one tensor affects the other.

```python
x = torch.tensor([[1., 2, 3], [4, 5, 6]])
# check if two tensors have the same storage
def same_storage(x: torch.Tensor, y: torch.Tensor):
    return x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
```

To get row 0:

```python
y = x[0] 
assert torch.equal(y, torch.tensor([1., 2, 3]))
assert same_storage(x, y)
```

To get column 1:

```python
y = x[:, 1] 
assert torch.equal(y, torch.tensor([2, 5]))
assert same_storage(x, y)
```

To view 2x3 matrix as 3x2 matrix:

```python
y = x.view(3, 2) 
assert torch.equal(y, torch.tensor([[1, 2], [3, 4], [5, 6]]))
assert same_storage(x, y)
```

To transpose the matrix:

```python
y = x.transpose(1, 0) 
assert torch.equal(y, torch.tensor([[1, 4], [2, 5], [3, 6]]))
assert same_storage(x, y)
```

To check that mutating x also mutates y.

```python
x[0][0] = 100
assert y[0][0] == 100
```

Note that some views are non-contiguous entries, which means that further views aren't possible.

```python
x = torch.tensor([[1., 2, 3], [4, 5, 6]]) 
y = x.transpose(1, 0) 
assert not y.is_contiguous()
try:
    y.view(2, 3)
    assert False
except RuntimeError as e:
    assert "view size is not compatible with input tensor's size and stride" in str(e)
```

One can enforce a tensor to be contiguous first:

```python
y = x.transpose(1, 0).contiguous().view(2, 3)  # @inspect y
assert not same_storage(x, y)
```

However, views are free, copying take both (additional) memory and compute.

#### Tensor Elementwise

These operations apply some operation to each element of the tensor and <u>return a (new) tensor of the same shape</u>.

```python
x = torch.tensor([1, 4, 9])
assert torch.equal(x.pow(2), torch.tensor([1, 16, 81]))
assert torch.equal(x.sqrt(), torch.tensor([1, 2, 3]))
assert torch.equal(x.rsqrt(), torch.tensor([1, 1 / 2, 1 / 3]))  # i -> 1/sqrt(x_i)
assert torch.equal(x + x, torch.tensor([2, 8, 18]))
assert torch.equal(x * 2, torch.tensor([2, 8, 18]))
assert torch.equal(x / 0.5, torch.tensor([2, 8, 18]))
```

`triu` takes the upper triangular part of a matrix.

```python
x = torch.ones(3, 3).triu()  # @inspect x
assert torch.equal(x, torch.tensor([
    [1, 1, 1],
    [0, 1, 1],
    [0, 0, 1]],
))
```

This is useful for computing an causal attention mask, where M[i, j] is the contribution of i to j.

#### Tensor Matmul

Finally, the bread and butter of deep learning: matrix multiplication.

```python
x = torch.ones(16, 32)
w = torch.ones(32, 2)
y = x @ w
assert y.size() == torch.Size([16, 2])
```

In general, we perform operations for every example in a batch and token in a sequence.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250526092145490.png" alt="image-20250526092145490" style="zoom:50%;" />

```python
x = torch.ones(4, 8, 16, 32)
w = torch.ones(32, 2)
y = x @ w
assert y.size() == torch.Size([4, 8, 16, 2])
```

In this case, we iterate over values of the first 2 dimensions of `x` and multiply by `w`.

### Tensor Einops

The traditional PyTorch code like:

```python
x = torch.ones(2, 2, 3)  # batch, sequence, hidden  
y = torch.ones(2, 2, 3)  # batch, sequence, hidden  
z = x @ y.transpose(-2, -1)  # batch, sequence, sequence  
```

Easy to mess up the dimensions (what is -2, -1?). We can use `tensor_einops` as a solution. Einops is a library for manipulating tensors where dimensions are named. It is inspired by Einstein summation notation. You can take [[Einops tutorial\]](https://einops.rocks/1-einops-basics/) as reference.

First, let's see how to eep track of tensor dimension. We can use jaxtyping to annotate the dim of the tensor.

```python
x: Float[torch.Tensor, "batch seq heads hidden"] = torch.ones(2, 2, 1, 3)
```

> Note: this is just documentation (no enforcement).

#### einops einsum

Einsum is **generalized matrix multiplication with good bookkeeping**. Let's see an example:

```python
# Define two tensors:
x: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2, 3, 4)  
y: Float[torch.Tensor, "batch seq2 hidden"] = torch.ones(2, 3, 4) 
# Old way:
z = x @ y.transpose(-2, -1)  # batch, sequence, sequence  
```

We can use new(einops) way:

```python
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
```

Dimensions that are not named in the output are summed over.

We can also use ``...`` to represent broadcasting over any number of dimensions:

```python
z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")
```

#### einops reduce

You can reduce a single tensor via some operation (e.g., sum, mean, max, min). let's use mean over tensor as an example:

```python
x: Float[torch.Tensor, "batch seq hidden"] = torch.ones(2, 3, 4)
# old way
y = x.mean(dim=-1)
```

The new way which use einops:

```python
y = reduce(x, "... hidden -> ...", "mean") 
```

#### einops rearrange

Sometimes, a dimension represents two dimensions and you want to operate on one of them. Let's see an example:

```python
x: Float[torch.Tensor, "batch seq total_hidden"] = torch.ones(2, 3, 8)
```

`total_hidden`  is a flattened representation of `heads * hidden1`. We can Break up `total_hidden` into two dimensions (`heads` and `hidden1`) using `rearrange`.

```python
x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)
```

We can combine `heads` and `hidden2` back together:

```python
x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")
```

### Tensor Operations Flops

Having gone through all the operations, let us examine their computational cost. We use flop-based definition to mesure it. 

A floating-point operation (FLOP) is a basic operation like addition (x + y) or multiplication (x y). There are wo terribly confusing acronyms (pronounced the same!):

- **FLOPs**: floating-point operations (measure of computation done)

- **FLOP/s**: floating-point operations per second (also written as FLOPS), which is used to measure the speed of hardware.

Let's see some real examples:

- Training GPT-3 (2020) took 3.14e23 FLOPs [[article\]](https://lambdalabs.com/blog/demystifying-gpt-3)
- Training GPT-4 (2023) is speculated to take 2e25 FLOPs  [[article\]](https://patmcguinness.substack.com/p/gpt-4-details-revealed)

- A100 has a peak performance of 312 teraFLOP/s  [[spec\]](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)
- H100 has a peak performance of 1979 teraFLOP/s with sparsity, 50% without  [[spec\]](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet)
- With 8 H100s for 2 weeks, total FLOPs is:

```python
total_flops = 8 * (60 * 60 * 24 * 7) * h100_flop_per_sec # 4.788e+21
```

#### Linear Model

As motivation, suppose you have a linear model.

- We have n points

- Each point is d-dimsional

- The linear model maps each d-dimensional vector to a k outputs

The code is showed as follows:

```python
if torch.cuda.is_available():
    B = 16384  # Number of points
    D = 32768  # Dimension
    K = 8192   # Number of outputs
else:
    B = 1024
    D = 256
    K = 64
device = get_device()
x = torch.ones(B, D, device=device)
w = torch.randn(D, K, device=device)
y = x @ w
```

We have one multiplication ($x[i][j] * w[j][k]$) and one addition per (i, j, k) triple. The FLOPs for linear model is:

```python
actual_num_flops = 2 * B * D * K
```

#### FLOPs of other operations

Elementwise operation on a m x n matrix requires `O(m n)` FLOPs.  Addition of two m x n matrices requires `mn` FLOPs. In general, no other operation that you'd encounter in deep learning is as expensive as matrix multiplication for large enough matrices.

We can interprete matmul in following way:

- B is the number of data points
- (D K) is the number of parameters
- FLOPs for forward pass is 2 (# tokens) (# parameters)

It turns out this generalizes to Transformers (to a first-order approximation).

How do our FLOPs calculations translate to wall-clock time (seconds)? Let us time it!

```python
def time_matmul(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the number of seconds required to perform `a @ b`."""
    # Wait until previous CUDA threads are done
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    def run():
        # Perform the operation
        a @ b
        # Wait until CUDA threads are done
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    # Time the operation `num_trials` times
    num_trials = 5
    total_time = timeit.timeit(run, number=num_trials)
    return total_time / num_trials

actual_time = time_matmul(x, w)  
actual_flop_per_sec = actual_num_flops / actual_time 
```

The result in my computer is:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250526112226188.png" alt="image-20250526112226188" style="zoom:60%;" />

Each GPU has a specification sheet that reports the peak performance: A100 [[spec\]](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf);H100 [[spec\]](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet). Note that the FLOP/s depends heavily on the data type!

#### Model FLOPs utilization (MFU)

The definition of MFU is **(actual FLOP/s) / (promised FLOP/s) [ignore communication/overhead]**. To calculate MFU, we can use following code:

```python
mfu = actual_flop_per_sec / promised_flop_per_sec
```

Usually, MFU >= 0.5 is quite good (and will be higher if matmuls dominate).

### Gradient

#### Gradients Basics

So far, we've constructed tensors (which correspond to either parameters or data) and passed them through operations (forward). Now, we're going to compute the gradient (backward).

As a simple example, let's consider the simple linear model: $y = 0.5 (x * w - 5)^2$. 

- Forward pass: compute loss

```python
x = torch.tensor([1., 2, 3])
w = torch.tensor([1., 1, 1], requires_grad=True)  # Want gradient
pred_y = x @ w
loss = 0.5 * (pred_y - 5).pow(2)
```

- Backward pass: compute gradients

```python
loss.backward()
assert loss.grad is None
assert pred_y.grad is None
assert x.grad is None
assert torch.equal(w.grad, torch.tensor([1, 2, 3]))
```

#### Gradients Flops

Let us do count the FLOPs for computing gradients. Let's use linear model as an example:

```python
if torch.cuda.is_available():
    B = 16384  # Number of points
    D = 32768  # Dimension
    K = 8192   # Number of outputs
else:
    B = 1024
    D = 256
    K = 64
device = get_device()
x = torch.ones(B, D, device=device)
w1 = torch.randn(D, D, device=device, requires_grad=True)
w2 = torch.randn(D, K, device=device, requires_grad=True)
# Model: x --w1--> h1 --w2--> h2 -> loss
h1 = x @ w1
h2 = h1 @ w2
loss = h2.pow(2).mean()
```

For the forward pass:

- Multiply $x[i][j] * w1[j][k]$

- Add to $h1[i][k]$

- Multiply $h1[i][j] * w2[j][k]$    

- Add to $h2[i][k]$

So the FLOPs is:

```python
num_forward_flops = (2 * B * D * D) + (2 * B * D * K)
```

For the backward pass, Recall model: x --w1--> h1 --w2--> h2 -> loss

- h1.grad = d loss / d h1

- h2.grad = d loss / d h2

- w1.grad = d loss / d w1

- w2.grad = d loss / d w2

We can focus on w2.grad: $w2.grad[j,k] = \sum_i h1[i,j] * h2.grad[i,k]$:

```python
assert w2.grad.size() == torch.Size([D, K])
assert h1.size() == torch.Size([B, D])
assert h2.grad.size() == torch.Size([B, K])
```

For each (i, j, k), multiply and add.

```python
num_backward_flops += 2 * B * D * K
```

We can do it for w1 (D*D parameters) as well (though don't need x.grad).

```python
num_backward_flops += (2 + 2) * B * D * D
```

There is a nice graphical visualization:  [[article\]](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4)

<img src="https://stanford-cs336.github.io/spring2025-lectures/var/files/image-c4037e492b7a4aa56d859f75b243c830-https_miro_medium_com_v2_resize_fit_1400_format_webp_1_VC9y_dHhCKFPXj90Qshj3w_gif" alt="img" style="zoom:50%;" />

Putting it togther:

- Forward pass: 2 (# data points) (# parameters) FLOPs

- Backward pass: 4 (# data points) (# parameters) FLOPs

- Total: **6 (# data points) (# parameters) FLOPs**

## Models

### Model Parameters

Model parameters are stored in PyTorch as `nn.Parameter` objects. We can confirm it using following example:

```python
input_dim = 16384
output_dim = 32
w = nn.Parameter(torch.randn(input_dim, output_dim))
assert isinstance(w, torch.Tensor)  # Behaves like a tensor
assert type(w.data) == torch.Tensor  # Access the underlying tensor
```

The output is:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250526123800260.png" alt="image-20250526123800260" style="zoom:50%;" />

Note that each element of `output` scales as `sqrt(input_dim)` like -84.09819030761719. Large values can cause gradients to blow up and cause training to be unstable.

We want an initialization that is invariant to `input_dim`. To do that, we simply rescale by `1/sqrt(input_dim)`. for example:

```python
w = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))
```

The result is:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250526124026347.png" alt="image-20250526124026347" style="zoom:50%;" />

Now each element of `output` is constant: -0.12920215725898743. This method is called Xavier initialization.  [[paper\]](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)[[stackexchange\]](https://ai.stackexchange.com/questions/30491/is-there-a-proper-initialization-technique-for-the-weight-matrices-in-multi-head).

To be extra safe, we truncate the normal distribution to [-3, 3] to avoid any chance of outliers.

```python
w = nn.Parameter(nn.init.trunc_normal_(torch.empty(input_dim, output_dim), std=1 / np.sqrt(input_dim), a=-3, b=3))
```

### Custom Models

Let's build up a simple deep linear model using `nn.Parameter`. 

```python
class Linear(nn.Module):
    """Simple linear layer."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight
class Cruncher(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            Linear(dim, dim)
            for i in range(num_layers)
        ])
        self.final = Linear(dim, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply linear layers
        B, D = x.size()
        for layer in self.layers:
            x = layer(x)
        # Apply final head
        x = self.final(x)
        assert x.size() == torch.Size([B, 1])
        # Remove the last dimension
        x = x.squeeze(-1)
        assert x.size() == torch.Size([B])
        return x
      
      
 def get_num_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())
```

Let's check the parameters:

```python
D = 64  # Dimension
num_layers = 2
model = Cruncher(dim=D, num_layers=num_layers)
param_sizes = [
    (name, param.numel())
    for name, param in model.state_dict().items()
]
assert param_sizes == [
    ("layers.0.weight", D * D),
    ("layers.1.weight", D * D),
    ("final.weight", D),
]
num_parameters = get_num_parameters(model)
assert num_parameters == (D * D) + (D * D) + D
```

Remember to move the model to the GPU.

```python
device = get_device()
model = model.to(device)
Run the model on some data.
B = 8  # Batch size
x = torch.randn(B, D, device=device)
y = model(x)
assert y.size() == torch.Size([B])
```

Finally we run the model on some data.

```python
B = 8  # Batch size
x = torch.randn(B, D, device=device)
y = model(x)
assert y.size() == torch.Size([B])
```

### Randomness

Randomness shows up in many places: parameter initialization, dropout, data ordering, etc. For reproducibility, we recommend you always pass in a different random seed for each use of randomness.

Determinism is particularly useful when debugging, so you can hunt down the bug. There are three places to set the random seed which you should do all at once just to be safe.

```python
# Torch
seed = 0
torch.manual_seed(seed)
# NumPy
import numpy as np
np.random.seed(seed)
# Python
import random
random.seed(seed)
```

### Data Loading

In language modeling, data is a sequence of integers (output by the tokenizer).It is convenient to serialize them as numpy arrays (done by the tokenizer).  We can save it to a npy file.

```python
orig_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
orig_data.tofile("data.npy")
```

You can load them back as numpy arrays.  Remember don't want to load the entire data into memory at once (LLaMA data is 2.8TB). Use `memmap` to lazily load only the accessed parts into memory.

```python
data = np.memmap("data.npy", dtype=np.int32)
assert np.array_equal(data, orig_data)
```

> Learn more about `memmap` here: https://numpy.org/doc/stable/reference/generated/numpy.memmap.html

A *data loader* generates a batch of sequences for training. Let's see how to implement data loader:

```python
def get_batch(data: np.array, batch_size: int, sequence_length: int, device: str) -> torch.Tensor:
    # Sample batch_size random positions into data.
    start_indices = torch.randint(len(data) - sequence_length, (batch_size,))
    assert start_indices.size() == torch.Size([batch_size])
    # Index into the data.
    x = torch.tensor([data[start:start + sequence_length] for start in start_indices])
    assert x.size() == torch.Size([batch_size, sequence_length])
    
    # By default, CPU tensors are in paged memory. We can explicitly pin.
    # # This allows us to copy x from CPU into GPU asynchronously.
    if torch.cuda.is_available():
        x = x.pin_memory()
    
    # This allows us to do two things in parallel (not done here):
    # - Fetch the next batch of data into CPU
    # Process x on the GPU.
    x = x.to(device, non_blocking=True)
    
    return x

B = 2  # Batch size
L = 4  # Length of sequence
x = get_batch(data, batch_size=B, sequence_length=L, device=get_device())
assert x.size() == torch.Size([B, L])
```

### Optimzer

Recall our deep linear model. 

```python
B = 2
D = 4
num_layers = 2
model = Cruncher(dim=D, num_layers=num_layers).to(get_device())
```

Let's define the [AdaGrad optimizer](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf). There are several optimize algorithms used by deep learning.

- **momentum** = SGD + exponential averaging of grad

- **AdaGrad** = SGD + averaging by grad^2

- **RMSProp** = AdaGrad + exponentially averaging of grad^2

- **Adam** = RMSProp + momentum

```python
class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        super(AdaGrad, self).__init__(params, dict(lr=lr))
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                # Optimizer state
                state = self.state[p]
                grad = p.grad.data
                # Get squared gradients g2 = sum_{i<t} g_i^2
                g2 = state.get("g2", torch.zeros_like(grad))
                # Update optimizer state
                g2 += torch.square(grad)
                state["g2"] = g2
                # Update parameters
                p.data -= lr * grad / torch.sqrt(g2 + 1e-5)
```

Let's see the interal state of optimizer:

```python
optimizer = AdaGrad(model.parameters(), lr=0.01)
state = model.state_dict()
```

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250526131148081.png" alt="image-20250526131148081" style="zoom:50%;" />

After the definition of optimizer, we can take a full at how to train a model:

- Compute gradient:

```python
x = torch.randn(B, D, device=get_device())
y = torch.tensor([4., 5.], device=get_device())
pred_y = model(x)
loss = F.mse_loss(input=pred_y, target=y)
loss.backward()
```

- Take a step

```python
optimizer.step()
```

- Free up the memory 

```python
optimizer.zero_grad(set_to_none=True)
```

The memory used in training part is showed as follows:

```python
# Parameters
num_parameters = (D * D * num_layers) + D  
assert num_parameters == get_num_parameters(model)
# Activations
num_activations = B * D * num_layers  
# Gradients
num_gradients = num_parameters  
# Optimizer states
num_optimizer_states = num_parameters  
# Putting it all together, assuming float32
total_memory = 4 * (num_parameters + num_activations + num_gradients + num_optimizer_states)
# flops
flops = 6 * B * num_parameters
```

### Training Loop

Let's put all things together to make a training loop:

```python
D = 16
true_w = torch.arange(D, dtype=torch.float32, device=get_device())
def get_batch(B: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(B, D).to(get_device())
    true_y = x @ true_w
    return (x, true_y)

def train(name: str, get_batch,
          D: int, num_layers: int,
          B: int, num_train_steps: int, lr: float):
    model = Cruncher(dim=D, num_layers=0).to(get_device())
    optimizer = SGD(model.parameters(), lr=0.01)
    for t in range(num_train_steps):
        # Get data
        x, y = get_batch(B=B)
        # Forward (compute loss)
        pred_y = model(x)
        loss = F.mse_loss(pred_y, y)
        # Backward (compute gradients)
        loss.backward()
        # Update parameters
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

#Let's do a basic run
train("simple", get_batch, D=D, num_layers=0, B=4, num_train_steps=10, lr=0.01)
```

### Checkpointing

Training language models take a long time and certainly will certainly crash. You don't want to lose all your progress. During training, it is useful to periodically save your model and optimizer state to disk.

```python
model = Cruncher(dim=64, num_layers=3).to(get_device())
optimizer = AdaGrad(model.parameters(), lr=0.01)
# Save the checkpoint:
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}
torch.save(checkpoint, "model_checkpoint.pt")
# Load the checkpoint:
loaded_checkpoint = torch.load("model_checkpoint.pt")
```

### Mixed Precision Training

 Choice of data type (float32, bfloat16, fp8) have tradeoffs.

- Higher precision: more accurate/stable, more memory, more compute

- Lower precision: less accurate/stable, less memory, less compute

How can we get the best of both worlds? The solutionis to use float32 by default, but use {bfloat16, fp8} when possible.

A concrete plan is:

- Use {bfloat16, fp8} for the forward pass (activations).

- Use float32 for the rest (parameters, gradients).

This is called mixed precision training [[Micikevicius+ 2017\]](https://arxiv.org/pdf/1710.03740.pdf)

We can see this in real industrial applications:

- Pytorch has an automatic mixed precision (AMP) library. https://pytorch.org/docs/stable/amp.html

https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/

- NVIDIA's Transformer Engine supports FP8 for linear layers. We can use FP8 pervasively throughout training  [[Peng+ 2023\]](https://arxiv.org/pdf/2310.18313.pdf)

