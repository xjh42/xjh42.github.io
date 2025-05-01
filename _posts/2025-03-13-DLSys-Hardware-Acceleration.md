---
  title: 'DLSys: Hardware Acceleration'
date: 2025-03-13 10:02:57
tags:
- hardware
categories:
- dlsys

pin: false
math: true
---

In this post, We'll introduce how to make model run faster. First we'll introduce the general acceleration techniques, then we'll introduce cuda programming, finally we'll implement the basics.

## General Acceleration Techniques

There are several layers in machine learning frameworks: 

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250313100921527.png" alt="image-20250313100921527" style="zoom:40%;" />

Our goal in this post is to optimize tensor linear libraries on different hardwares.

### Vectorization

Vectorization let us compute single instruction on multiple data(SIMD) in one cycle. For example, the following code add two arrays of length 256 using vectorization.

```
void vecadd(float* A, float *B, float* C) {
  for (int i = 0; i < 64; ++i) {
    float4 a = load_float4(A + i*4);
    float4 b = load_float4(B + i*4);
    float4 c = add_float4(a, b);
    store_float4(C + i* 4, c);
  }
}
```

In vectorization, there consist of aligment requirements:  memory (A, B, C) needs to be aligned to 128 bits. Many array library provides aligment malloc operations.

> For numpy reference: https://github.com/numpy/numpy/issues/5312

### Data layout and strides

Next, we should consider how to store a matrix in memory. The memory is one continuous array, so we need consider how to convert matrix to an array. There several techniques:

- Row major: `A[i, j] => Adata[i * A.shape[1] + j]`
- Column major: `A[i, j] => Adata[j * A.shape[0] + i]`
- Strides format: `A[i, j] => Adata[i * A.strides[0] + j * A.strides[1]] `

Stride format is a more general format which can convert to either row major or column major by changing the `strides`. 

The advantages to use strides is we can **perform transformation/slicing in zero copy way**

- **Slice**: change the begin offset and shape
- **Transpose**: swap the strides
- **Broadcast**: insert a stride equals 0

The disadvantages is **memory access becomes not continuous**

- Makes vectorization harder
- Many linear algebra operations may require compact the array first

> `numpy.ascontiguousarray`: https://numpy.org/doc/2.2/reference/generated/numpy.ascontiguousarray.html

### Parallelization

We can also run multiple tasks on multiple hardware threads to speed up the computation. The following examples using openmp to apply parallelization:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250313191523054.png" alt="image-20250313191523054" style="zoom:50%;" />

`#pragma omp parallel for` to change for loop to a parallelized program.

### Case Study: Matrix Multiplication 

Just like `numpy`, matrix multiplication is to compute: `C = dot(A, B.T)`. We can quickly write a vanilla matrix multiplication program:

```c++
float A[N][N], B[N][N], C[N][N];
for (int i = 0; i < N; i++) {
  for (int j = 0; j < N; j++) {
    C[i][j] = 0;
    for (int k = 0; k < n; k++) {
      C[i][j] += A[i][k] * B[j][k];
    }
  }
}
```

Obviously, the run time is O(n^3). If we think about the memory hierarchy, the cost is even worse. The common memory hierarchy is showed as follows:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250313192539652.png" alt="image-20250313192539652" style="zoom:50%;" />

#### Architecture aware analysis

Let rethink the cost for vanilla matrix multiplication:

```c++
dram float A[N][N], B[N][N], C[N][N];
for (int i = 0; i < N; i++) {
  for (int j = 0; j < N; j++) {
    register float c = 0;
    for (int k = 0; k < n; k++) {
      register float a = A[i][k];
      register float b = B[j][k];
      c += a * b;
    }
    C[i][j] = c;
  }
}
```

- **cost analysis**:
  - A’s dram->register time cost: n^3
  -  B’s dram->register time cost: n^3 
  - A’s register memory cost : 1
  -  B’s register memory cost : 1 
  - C’s register memory cost : 1

- **Load cost**: 2 * dramspeed * n^3
- **Register cost**: 3

#### Register tiled matrix multiplication

In order to reduce the loading cost from dram to register, we can load a sub-matrix to registers in a single time and compute the register matrix multiplication:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250313194614876.png" alt="image-20250313194614876" style="zoom:50%;" />

```c++ 
dram float A[N/V1][N/V3][V1][V3];
dram float B[N/V2][N/V3][V2][V3];
dram float C[N/V1][N/V2][V1][V2];

for (int i = 0; i < N / V1; i++) {
  for (int j = 0; j < N / V2; j++) {
    register c[V1][V2] = 0;
    for (int k = 0; k < N / V3; k++) {
      register a[V1][V3] = A[i][k];
      register b[V2][V3] = A[j][k];
      c += dot(a, b.T);
    }
    C[i][j] = c;
  }
}
```

- Cost analysis:

  A’s dram->register time cost: n^3/v2 

  B’s dram->register time cost: n^3/v1 

  A’s register memory cost: v1*v3 

  B’s register memory cost: v2*v3

   C’s register memory cost: v1*v2

- load cost: dramspeed * (n^3/v2 + n^3/v1)
- Register cost: v1*v3 + v2*v3 + v1*v2

However, the number of registers in CPU is small, so v1/v2/v3 cannot be very big. So the improvement is not very efficient.

#### Cache line aware tiling

We can avoid the limitation of the number of registers is small by place the sub-matrix to l1cache.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250313194817570.png" alt="image-20250313194817570" style="zoom:40%;" />

```c++
dram A[N/B1][B1][N];
dram B[N/B2][B2][N];
dram C[N/B1][N/B2][B1][B2];
for (int i = 0; i < N/B1; i++) {
  l1cache a[B1][N] = A[i];
  for (int j = 0; j < N/B2; j++) {
    l1cache b[B2][N] = B[i];
    
    // Sub-procedure, can apply register tiling here
    C[i][j] = dot(a, b.T);
  }
}
```

- cost analysis 

  A’s dram->l1 time cost: n^2 

  B’s dram->l1 time cost: n^3 / b1

- Constraints: 

  - `b1 * n + b2 * n < l1 cache size`

  - To still apply register blocking on dot: `b1 % v1 == 0` , `b2 % v2 == 0`

#### Cache Line Tiling + Register Tiling

For cache line tiling, we can do more. We can apply register tiling in the computation of `C[i][j] = dot(a, b.T)`

```c++
dram A[N/B1][B1/V1][N][V1];
dram B[N/B2][B2/V2][N][V2];
dram C[N/B1][N/B2][B1/V1][B2/V2][V1][V2];
for (int i = 0; i < N/B1; i++) {
  l1cache a[B1/V1][N][V1] = A[i];
  for (int j = 0; j < N/B2; j++) {
    l1cache b[B2/V2][N][V2] = B[i];
    
    // Sub-procedure, can apply register tiling here
    for (int x = 0; x < B1/V1; x++) {
      for (int y = 0; y < B2/V2; y++) {
        register c[v1][v2] = 0;
        for (int k = 0; k < N; k++) {
          register float ar[V1] = a[x][k];
          register float br[v2] = b[y][k];
          c += dot(ar, br.T);
        }
        C[i][j][x][y] = c;
      }
    }
  }
}
```

Load cost: 

- dram->l1cache: N^3/B1 + N^2
- L1cache->register: N^3 / V1 + N^3/V2

Register to use: V1 * V2 + V1 + V2

#### Memory Reuse Pattern

The reason why register tiling or cache line tiling is efficient is we can reuse memory. For eaxmple, for register tiling:

```c++
dram float A[N/V1][N/V3][V1][V3];
dram float B[N/V2][N/V3][V2][V3];
dram float C[N/V1][N/V2][V1][V2];

for (int i = 0; i < N / V1; i++) {
  for (int j = 0; j < N / V2; j++) {
    register c[V1][V2] = 0;
    for (int k = 0; k < N / V3; k++) {
      register a[V1][V3] = A[i][k];
      register b[V2][V3] = A[j][k];
      c += dot(a, b.T);
    }
    C[i][j] = c;
  }
}
```

A get reused V2 times and B get reused V1 times. So the runtime for load A is **n^3/v2**; the runtime for load B is **n^3/v1**.

We can conclude the common reuse pattern for matrix multiplication:

```
float A[n][n];
float B[n][n];
float C[n][n];
C[i][j] = sum(A[i][k] * B[j][k], axis=k)
```

Access of A is independent of j, tile the j dimension by v enables reuse of A for v times.

## CUDA Programming

### GPU Programming

So What is GPU? According to the definition from Nvidia:The GPU is specialized for **highly parallel computations** and therefore designed such that more transistors are devoted to data processing rather than data caching and flow control. 

The difference in capabilities between the GPU and the CPU exists because they are designed with different goals in mind. While the CPU is designed to **excel at executing a sequence of operations**, called a *thread*, as fast as possible and can execute a few tens of these threads in parallel, the GPU is designed to excel at executing thousands of them in parallel (amortizing the slower single-thread performance to achieve greater throughput).

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250315111152762.png" alt="image-20250315111152762" style="zoom:50%;" />

GPU programming use **SIMT(Single instruction multiple threads)** programming mode. It means <u>all threads executes the same code, but can take different path</u>.

Threads are grouped into **blocks**(Thread within the same block have *shared memory*). Blocks are grouped into a **launch grid**. A kernel in GPU executes a grid.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250315111808977.png" alt="image-20250315111808977" style="zoom:33%;" />

Let's look a simple example: add two vectors and store the result to a new vector. The cpu version code is showed as follows:

```c++
#include <iostream>
#include <chrono>
#include <limits.h>

#define N 10000000

void VecAddCPU(float *A, float *B, float *C, int n) {
  for (int i = 0; i < n; i++) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  float A[N], B[N], C[N];
  for (int i = 0; i < N; i++) {
    A[i] = 100.0;
    B[i] = 100.0;
  }

  auto start_time = std::chrono::high_resolution_clock::now();
  VecAddCPU(A, B, C, N);
  auto end_time = std::chrono::high_resolution_clock::now();
  auto time = end_time - start_time;
  std::cout << "VecAddCPU" << " took " <<
    time/std::chrono::milliseconds(1) << "ms to run.\n";
}
```

The output in my computer is:

```
VecAddCPU took 37ms to run.
```

If we reimplement the program using cuda . It consists of two parts:

- Kernel function to compute a single add:

```c++
__global__ void VecAddKernel(float *A, float *B, float *C, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}
```

`blockDim` is the dimension for block, `blockIdx` is the idx of block the kernel will run, `threadIdx` is the idx in the block. We can use these global variables to compute the global offset.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250315183533026.png" alt="image-20250315183533026" style="zoom:50%;" />

- host side: to allocate device memory and launch the grid of computation:

```c++
void VecAddCUDA(float *Acpu, float *Bcpu, float *Ccpu, int n) {
  float *dA, *dB, *dC;

  cudaMalloc(&dA, n * sizeof(float));
  cudaMalloc(&dB, n * sizeof(float));
  cudaMalloc(&dC, n * sizeof(float));

  cudaMemcpy(dA, Acpu, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, Bcpu, n * sizeof(float), cudaMemcpyHostToDevice);

  int thread_per_block = 512;
  int nblocks = (n +  thread_per_block - 1) / thread_per_block;
  VecAddKernel<<<nblocks, thread_per_block>>>(dA, dB, dC, n);
  cudaMemcpy(Ccpu, dC, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}
```

The full program which contains initialization is showed as follows:

```c++
#include <iostream>
#include <chrono>
#include <limits.h>

#define N 10000000

__global__ void VecAddKernel(float *A, float *B, float *C, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

void VecAddCUDA(float *Acpu, float *Bcpu, float *Ccpu, int n) {
  float *dA, *dB, *dC;

  cudaMalloc(&dA, n * sizeof(float));
  cudaMalloc(&dB, n * sizeof(float));
  cudaMalloc(&dC, n * sizeof(float));

  cudaMemcpy(dA, Acpu, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, Bcpu, n * sizeof(float), cudaMemcpyHostToDevice);

  int thread_per_block = 512;
  int nblocks = (n +  thread_per_block - 1) / thread_per_block;
  VecAddKernel<<<nblocks, thread_per_block>>>(dA, dB, dC, n);
  cudaMemcpy(Ccpu, dC, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}

int main() {
  float A[N], B[N], C[N];
  for (int i = 0; i < N; i++) {
    A[i] = 100.0;
    B[i] = 100.0;
  }

  auto start_time = std::chrono::high_resolution_clock::now();
  VecAddCUDA(A, B, C, N);
  auto end_time = std::chrono::high_resolution_clock::now();
  auto time = end_time - start_time;
  std::cout << "VecAddCUDA" << " took " <<
    time/std::chrono::milliseconds(1) << "ms to run.\n";
}
```

The output in my computer is:

```c++
VecAddCUDA took 1ms to run.
```

Use cuda can archive 37x speed up for this simple case!

### GPU memory hierarchy

If you are curious about the structure for cuda programming mode: grid->block->thread, you should take a look at the gpu memory hierarchy:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250315184402084.png" alt="image-20250315184402084" style="zoom:50%;" />

The access time decreases from register, shared memory, and global memory. Like the previous example: matrix multiplication, we should use the **reuse pattern** to reduce data movement cost from global memory -> shared memory-> registers.

Let's use **window sum** as an example to show how to apply **reuse pattern** to reduce data movement.

Like `VecAdd` example, we can implement it using following code:

```c++
#include <iostream>
#include <chrono>
#include <limits.h>

#define RADIUS 256
#define N 1000000000
#define THREADS_PER_BLOCK 512


__global__ void WindowSumSimpleKernel(float *A, float *B, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n) {
    float sum = 0;
    for (int dx = -RADIUS; dx <= RADIUS; dx++) {
      sum += A[dx + idx + RADIUS];
    }
    B[idx] = sum;
  }
}

void WindowSumSimple(float *Acpu, float *Bcpu, int n) {
  float *dA, *dB;

  cudaMalloc(&dA, (n + 2 * RADIUS) * sizeof(float));
  cudaMalloc(&dB, n * sizeof(float));

  cudaMemcpy(dA, Acpu, (n + 2 * RADIUS) * sizeof(float), cudaMemcpyHostToDevice);

  int thread_per_block = THREADS_PER_BLOCK;
  int nblocks = (n +  thread_per_block - 1) / thread_per_block;
  WindowSumSimpleKernel<<<nblocks, thread_per_block>>>(dA, dB, n);
  cudaMemcpy(Bcpu, dB, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(dA);
  cudaFree(dB);
}

int main() {
  float A[N + 2 * RADIUS], B[N];
  for (int i = 0; i < N + 2 * RADIUS; i++) {
    A[i] = 100.0;
  }

  auto start_time = std::chrono::high_resolution_clock::now();
  WindowSumSimple(A, B, N);
  auto end_time = std::chrono::high_resolution_clock::now();
  auto time = end_time - start_time;
  std::cout << "WindowSumSimple" << " took " <<
    time/std::chrono::microseconds(1) << " microsecond to run.\n";
}

```

The output in my computer is:

```
WindowSumSimple took 388 microsecond to run.
```

If we see the computation in details, we can find the reuse pattern:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250315234627463.png" alt="image-20250315234627463" style="zoom:50%;" />

An item in the middle location will be load multiple times in different threads. So we can load the items needed to shared memory to apply reuse pattern.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250316000534344.png" alt="image-20250316000534344" style="zoom:50%;" />

```c++
#include <iostream>
#include <chrono>
#include <limits.h>

#define RADIUS 256
#define N 1000000000
#define THREADS_PER_BLOCK 512


__global__ void WindowSumSharedKernel(float *A, float *B, int n) {
  __shared__ float temp[THREADS_PER_BLOCK + 2 * RADIUS];
  int base = blockDim.x + blockIdx.x;
  int out_idx = base + threadIdx.x;
  if (out_idx < n) {
    temp[threadIdx.x] = A[out_idx];
  }
  if (threadIdx.x < 2 * RADIUS) {
    temp[threadIdx.x + THREADS_PER_BLOCK] = A[out_idx + THREADS_PER_BLOCK];
  }
  __syncthreads();
  if (out_idx < n) {
    float sum = 0;
    for (int dx = -RADIUS; dx <= RADIUS; dx++) {
      sum += temp[dx + threadIdx.x + RADIUS];
    }
    B[out_idx] = sum;
  }
  
}


void WindowSumShared(float *Acpu, float *Bcpu, int n) {
  float *dA, *dB;

  cudaMalloc(&dA, (n + 2 * RADIUS) * sizeof(float));
  cudaMalloc(&dB, n * sizeof(float));

  cudaMemcpy(dA, Acpu, (n + 2 * RADIUS) * sizeof(float), cudaMemcpyHostToDevice);

  int thread_per_block = THREADS_PER_BLOCK;
  int nblocks = (n +  thread_per_block - 1) / thread_per_block;
  WindowSumSharedKernel<<<nblocks, thread_per_block>>>(dA, dB, n);
  cudaMemcpy(Bcpu, dB, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(dA);
  cudaFree(dB);
}

int main() {
  float A[N + 2 * RADIUS], B[N];
  for (int i = 0; i < N + 2 * RADIUS; i++) {
    A[i] = 100.0;
  }

  auto start_time = std::chrono::high_resolution_clock::now();
  WindowSumShared(A, B, N);
  auto end_time = std::chrono::high_resolution_clock::now();
  auto time = end_time - start_time;
  std::cout << "WindowSumSimple" << " took " <<
    time/std::chrono::microseconds(1) << " microsecond to run.\n";
}

```

The output in my computer is:

```
WindowSumSimple took 251 microsecond to run.
```

We can archive 1.54x speed up compare to the simple version.

### Case study: matrix multiplication on GPU

#### Thread-level: register tiling

We can write the kernel function as follows:

```c++
__global__ void mm(float A[N][N], float B[N][N], float C[N][N]) {
  int ybase = blockIdx.y * blockDim.y + threadIdx.y;
  int xbase = blockIdx.x * blockDim.x + threadIdx.x;
  float c[V][V] = {0};
  float a[V], b[V];
  for (int k = 0; k < N; ++k) {
    a[:] = A[k, ybase*V : ybase*V + V];
    b[:] = B[k, xbase*V : xbase*V + V];
    for (int y = 0; y < V; ++y) {
       for (int x = 0; x < V; ++x) {
         c[y][x] += a[y] * b[x];
       }
     }
   }
   C[ybase * V : ybase*V + V, xbase*V : xbase*V + V] = c[:];
}
```

The computation is showed as follows:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250319093818842.png" alt="image-20250319093818842" style="zoom:33%;" />

#### Block-level: shared memory tiling

To reduce memory access, we can preload a block of matrix to shared memory.

```c++
__global__ void mm(float A[N][N], float B[N][N], float C[N][N]) {
  __shared__ float sA[S][L], sB[S][L];
  float c[V][V] = {0};
  float a[V], b[V];
  int yblock = blockIdx.y;
  int xblock = blockIdx.x;
  for (int ko = 0; ko < N; ko += S) {
    __syncthreads();
    // needs to be implemented by thread cooperative fetching
    sA[:, :] = A[k : k + S, yblock * L : yblock * L + L];
    sB[:, :] = B[k : k + S, xblock * L : xblock * L + L];
    t
    // thread level pre-fetching
    for (int ki = 0; ki < S; ++ ki) {
      a[:] = sA[ki, threadIdx.y * V : threadIdx.y * V + V];
      b[:] = sA[ki, threadIdx.x * V : threadIdx.x * V + V];
      for (int y = 0; y < V; ++y) {
        for (int x = 0; x < V; ++x) {
          c[y][x] += a[y] * b[x];
        }
      }
    }
   }
   int ybase = blockIdx.y * blockDim.y + threadIdx.y;
   int xbase = blockIdx.x * blockDim.x + threadIdx.x;
   C[ybase * V : ybase*V + V, xbase*V : xbase*V + V] = c[:];
}
```

The computation is showed as follows:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250319094117479.png" alt="image-20250319094117479" style="zoom:33%;" />

By using block-level tiling, we can reduce memory access. 

- global->shared copy: **2 * N^3 / L**
- shared->register: **2 * N^3 / V**

The implementation of **thread cooperative fetching** is to convert:

```
sA[:, :] = A[k : k + S, yblock * L : yblock * L + L]; 
```

to real code:

```c++
int nthreads = blockDim.y * blockDim.x;
int tid = threadIdx.y * blockDim.x + threadIdx.x;
for(int j = 0; j < L * S / nthreads; ++j) {
  int y = (j * nthreads + tid) / L;
  int x = (j * nthreads + tid) % L;
  s[y, x] = A[k + y, yblock * L + x];
}
```

### More GPU Techniques

There are more GPU techniques:

- Global memory continuous read 
- Shared memory bank conflict 
- Software pipelining 
- Warp level optimizations 
- Tensor Core

We'll expand the topic later.

## Implementation

https://colab.research.google.com/drive/1OGj8xwwUsdsRulaYXU3TtJLWmgdo5VDl?usp=sharing
