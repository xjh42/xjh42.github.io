---
title: 'Language Models 3: The Evolution of Transformer-based Language Models'
date: 2025-04-31 09:52:23
tags:
- architecture
categories:
- language models
pin: false
math: true
---

In this. post, we'll first introduce the evolution of transformer-based language models, then introduce one of the important architecture--MOE(Mix-Of-Expert). After reading this post, you can have a better understanding of  the architecture of modern large language model like GPT4/Llama 4 etc.

## Evolution of Language Model

In this section, we'll take a detailed look at the evolution of the transformer-based language model. We'll take a quick recap of the ‘standard’ transformer. Then we'll spend most of time study what are common variations to the architecture / training process.

### Transformer Architecture

The following pciture shows the  original transformer by [[Ashish Vaswani](https://arxiv.org/abs/1706.03762)]:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250527103622544.png" alt="image-20250527103622544" style="zoom:50%;" />

You can learn more about it in [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) Apart from attention mechanism, the key point is showed as follows:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250527104033162.png" alt="image-20250527104033162" style="zoom: 33%;" />

However, modern variant transformer is decoder-only transformer. It's showed as follows:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250527104314380.png" alt="image-20250527104314380" style="zoom:50%;" />

The key difference between modern transformer and original transformer is listed as follows:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250527104455227.png" alt="image-20250527104455227" style="zoom:40%;" />

The following section will discuss them in detail.

### Variants of Transformer

The variants of modern language model is showed as follows. We will talk through many major architecture and hyperparameter variants. We'll focus on following questions:

- What do all these models have in common ?
- What parts vary? 
- What can we learn from this?

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250527131354104.png" alt="image-20250527131354104" style="zoom:80%;" />

#### **Norm Position**

First, let's see how to place layer norm to the architecture. In the original paper, they use post-norm which place norm after residual calculations. However, almost all the model use pre-norm which place norm before residual calculations.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250527132406673.png" alt="image-20250527132406673" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250527132439076.png" alt="image-20250527132439076" style="zoom:50%;" />

Pre-norm set up LayerNorm so that it doesn’t affect the main residual signal path which make training more stable. We can confirm it from expriments:

- From [[Salazar and Ngyuen 2019](https://arxiv.org/pdf/1910.05895)], we can see that using pre-norm can make model more accurate even without warm up setup:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250527132841076.png" alt="image-20250527132841076" style="zoom:50%;" />

- From [[Xiong 2020](https://arxiv.org/pdf/2002.04745)], we can see that using pre-norm can make model more accurate and the training process more stable.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250527133109098.png" alt="image-20250527133109098" style="zoom:50%;" />

- From [[Xiong 2020](https://arxiv.org/pdf/2002.04745)], we can give a explanations for why pre-norm works. Pre-norm make the gradient stable when layer grows without warm up. 

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250527133702998.png" alt="image-20250527133702998" style="zoom:50%;" />

Currently modern model(Grok, Gemma 2. Olmo 2) do "double" norm which make training more stable.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250527133842350.png" alt="image-20250527133842350" style="zoom:50%;" />

#### **Norm Type**

Original transformer use **LayerNorm**  which normalizes the mean and variance across $d_{model}$. The formula is:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250527134011296.png" alt="image-20250527134011296" style="zoom:50%;" />

Notable model like GPT3/2/1, OPT, GPT-J, BLOOM use layer norm.

Many modern LMs use **RMSNorm** which does not subtract mean or add a bias term:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250527134110592.png" alt="image-20250527134110592" style="zoom:33%;" />

Notable models like LLaMA-family, PaLM, Chinchilla, T5 use RMSNorm.

Why more modern models use RMSNorm ? 

- Compare to LayerNorm, it’s faster. 
  - **Fewer operations** (no mean calculation)
  - **Fewer parameters** (no bias term to store)
- Just good as LayerNorm

Norm computation really matters. From [[Ivanov et al 2023](https://arxiv.org/pdf/2007.00072)], 

- Matrix multiplies are the *vast* majority of FLOPs (and memory

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528000116905.png" alt="image-20250528000116905" style="zoom:100%;" />

- But FLOPs are not runtime, we can see from following table that normalization count 25% portion:

![image-20250528000406170](https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528000406170.png)

The data movement is really matters. We can see from following picture that nromalization data movement is bigger compare to the FLOPs.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528002614255.png" alt="image-20250528002614255" style="zoom:50%;" />

So RMSNorm can still matter due to the reduction  of *data movement*. From [[Narang et al 2021](https://arxiv.org/pdf/2102.11972)], RMSNorm runtime and surprisingly accuration gains have been seen in papers

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528003358642.png" alt="image-20250528003358642" style="zoom:80%;" />

#### Dropping Bias Terms

Compare to original transformer, most modern transformers don’t have bias terms.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528003637346.png" alt="image-20250528003637346" style="zoom:50%;" />

Similar to RMSnorm, the reasons why they do that is  memory  and optimization stability.

#### Activations

There are a whole zoo of activations: **ReLU, GeLU, Swish, ELU, GLU, GeGLU, ReGLU, SeLU, SwiGLU, LiGLU**.We'll take a good look at them.

- **ReLU**: the definition is showed as follows. It's used by original transformer, T5, Gopher, Chinchilla, OPT.

  <img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528090218744.png" alt="image-20250528090218744" style="zoom:50%;" />

- **GeLU**: the definition is showed as follows. It's used by GPT1/2/3, GPTJ, GPT-Neox, BLOOM.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528090327401.png" alt="image-20250528090327401" style="zoom:50%;" />

- **Gated activations (*GLU)**: GLUs modify the ‘first part’ of a FF layer. Instead of a linear + ReLU, augment the above with an (entrywise) linear term.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528090456029.png" alt="image-20250528090456029" style="zoom:50%;" />

This gives the gated variant (ReGLU) (note that we have an extra parameter (V))

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528090554791.png" alt="image-20250528090554791" style="zoom:50%;" />

There are some variants of gated activations. You can see them from following slides:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528090804322.png" alt="image-20250528090804322" style="zoom:30%;" />

> Note: Gated models use smaller dimensions for the $d_{ff}$ by 2/3

From [[Shazeer 2020](https://arxiv.org/abs/2002.05202)] and [[Narang et al 2020](https://arxiv.org/pdf/2102.11972)], we can know that *GLU works:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528091045136.png" alt="image-20250528091045136" style="zoom:40%;" />

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528091105067.png" alt="image-20250528091105067" style="zoom:40%;" />

#### Serial vs Parallel layers

Normal transformer blocks are serial. They compute attention, then the MLP. Could we parallelize the transformer block?

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528091248791.png" alt="image-20250528091248791" style="zoom: 50%;" />

A few models (GPTJ, PaLM, GPT-NeoX) do parallel layers. [[PaLM Paper](https://arxiv.org/pdf/2204.02311v5)] explains why they do parallel layers:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528091912000.png" alt="image-20250528091912000" style="zoom:67%;" />

If implemented right, LayerNorm can be shared, and matrix multiplies can be fused. Then we can get a faster training speed. Recently **Cohere Command A, Falcon 2 11B and Command R+** also use parallel layers to train model.

#### Position Embeddings

In the original transformer paper, it uses **sine embeddings** as position enbeddings which <u>add sines and cosines that enable localization</u>.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528092548654.png" alt="image-20250528092548654" style="zoom:60%;" />

There are some variants of position embeddings:

- **Absolute embeddings**: it adds a position vector to the embedding and is used by GPT1/2/3, OPT.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528092734497.png" alt="image-20250528092734497" style="zoom:40%;" />

- **Relative embeddings**: it adds a vector to the attention computation and is used by T5, Gopher, Chinchilla.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528092857539.png" alt="image-20250528092857539" style="zoom:40%;" />

But the most popular position embedding method: **Rope embeddings **(rotary position embeddings) which is used by GPTJ, PaLM, LLaMA and most 2024+ models.

In order to understand **Rope**, let's make a high level thought process: a relative position embedding should be some 𝑓(𝑥, 𝑖) s.t.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528093215566.png" alt="image-20250528093215566" style="zoom:33%;" />

That is, the attention function only gets to depend on the relative position (i-j). How do existing embeddings not fulfill this goal?

- Sine: Has various cross-terms that are not relative: 

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528093345717.png" alt="image-20250528093345717" style="zoom:33%;" />

- Absolute: obviously not relative
- Relative embeddings: it is not an inner product

How can we solve this problem?  We want our embeddings to be invariant to absolute position. We know that **inner products** are invariant to arbitrary rotation.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528093542595.png" alt="image-20250528093542595" style="zoom:40%;" />

[[Su et al 2021](https://arxiv.org/abs/2104.09864)] first introduce **Rope** to solve the problem. It pairs up the coordinates and rotate them in 2d.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528093821620.png" alt="image-20250528093821620" style="zoom:33%;" />

The math formula of **Rope** is: 

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528093937110.png" alt="image-20250528093937110" style="zoom:33%;" />

It actually apply **Rope** to query and key values. The implementation is showed as follows:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528094200408.png" alt="image-20250528094200408" style="zoom:40%;" />

> Rope embedding at each attention operation is to enforce position invariance

#### **Hyperparameters that matter**

In this section we'll answer following questions:

- How much bigger should the feedforward size be compared to hidden size? 
- How many heads, and should num_heads always divide hidden size? 
- What should my vocab size be? 
-  Do people even regularize these huge LMs? 
- How do people scale these models - very deep or very wide?

##### FFN  Model dimension ratio

FFN model is computed by following equation:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528094633418.png" alt="image-20250528094633418" style="zoom:33%;" />

There are two dimensions that are relevant – the feedforward dim ($d_{ff}$) and model dim ($d_{model}$). What should their relationship be? The answer is:
$$
d_{ff} = 4d_{model}
$$
This is almost always true. There’s just a few exceptions.

- Exception #1 – **GLU variants**:  GLU variants scale down by 2/3. This means most GLU variants have $d_{ff} = \frac{8}{3}d_{model}$.This is mostly what happens. Some notable such examples:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528095007313.png" alt="image-20250528095007313" style="zoom:50%;" />

- Exception #2 – **T5**： As we have (and will) see, most LMs are have boring, conservative hyperparameters. One exception is T5 [[Raffel et al 2020](https://arxiv.org/pdf/1910.10683)] which has some very bold settings. In particular, for the 11B model, they set 

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528095230027.png" alt="image-20250528095230027" style="zoom:33%;" />

For an astounding 64-times multiplier. They give a explanation in the paper:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528095344479.png" alt="image-20250528095344479" style="zoom:60%;" />

##### Head-dim*num-heads to model-dim ratio.

It comes from the multihead attention computation:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528095707725.png" alt="image-20250528095707725" style="zoom:50%;" />

But this doesn’t have to be true: we can have **head-dimensions > model-dim / num-heads**. Most models do follow this guideline.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528095750923.png" alt="image-20250528095750923" style="zoom:40%;" />

##### Aspect ratios

Aspect ratio means $d_{model}/d_{layer}$. Should my model be deep or wide? How deep and how wide? Most models are surprisingly consistent on this one too! 

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528095953373.png" alt="image-20250528095953373" style="zoom:50%;" />

From [[Tay et al 2021](https://arxiv.org/pdf/2109.10686)], we can know that extremely deep models are harder to parallelize and have higher latency:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528100326719.png" alt="image-20250528100326719" style="zoom:40%;" />

##### Vocabulary Sizes 

There are some practical design rules for choosing vocabulary size:

- **Monolingual models** – 30-50k vocab

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528125821603.png" alt="image-20250528125821603" style="zoom:50%;" />

- **Multilingual / production systems**: 100-250k

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528125951685.png" alt="image-20250528125951685" style="zoom:50%;" />

##### Dropout and other regularization

Do we need regularization during pretraining? There is *a lot* of data (trillions of tokens), more than parameters. SGD only does a single pass on a corpus (hard to memorize). This is all quite reasonable. but what do people do in practice?

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528130256848.png" alt="image-20250528130256848" style="zoom:50%;" />

In practice, many older models used dropout during pretraining. However, newer models (except Qwen) rely only on weight decay.

Why we need weight decay in LLM? [[Andriushchenko et al 2023](https://arxiv.org/pdf/2310.04415)] has interesting observations about LLM weight decay:

- Weight decay is not to control overfitting:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528131116620.png" alt="image-20250528131116620" style="zoom:50%;" />

- Weight decay interacts with learning rates (cosine schedule ) helps reduce training loss

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528131246949.png" alt="image-20250528131246949" style="zoom:50%;" />

#### **Stability tricks**

Recently, lots of attention on *stable training* because traing LLM is easy to be unstable.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528131546514.png" alt="image-20250528131546514" style="zoom:50%;" />

> Note: don’t train models that look like the blue curve!

The reason why training LLM is easy to be unstable is that there are **softmaxes** in LLM computation. **Softmaxes** can be ill-behaved due to exponentials / divison by zero.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528131820276.png" alt="image-20250528131820276" style="zoom:33%;" />

There are two softmax computation: output computation/attention computation. There are some tricks to make training process more stable: **z-loss**, **QK norm**, **Logit soft-capping**

#### **Output softmax stability – the z-loss**

Recall the logsoftmax computation:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528133021369.png" alt="image-20250528133021369" style="zoom:33%;" />

 [[Vincent 2016](https://arxiv.org/abs/1604.08859)] introduce the z-loss:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528133236978.png" alt="image-20250528133236978" style="zoom:33%;" />

This is useful for stability. [[PaLM](https://arxiv.org/abs/2204.02311)] pioneered this ‘z loss’ trick:

> The model is trained with the standard language modeling loss function, which is the average log probability of all tokens without label smoothing. We additionally use an auxiliary loss of $z_{loss} = 10^{-4}log^2Z$ to encourage the softmax normalizer log(Z) to be close to 0, which we found increases the stability of training.

##### **Attention softmax stability – the QK norm**

We can apply layer (RMS) norm to the query and keys  before going into the softmax operation.

![image-20250531215035674](https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531215035674.png)

##### **Logit soft-capping**

We can apply **soft-capping** to the logits to some maximum value via Tanh:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528133849656.png" alt="image-20250528133849656" style="zoom:50%;" />

However,  logit soft-capping prevents logits from blowing up, but also might have perf issues:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528133952479.png" alt="image-20250528133952479" style="zoom:70%;" />

#### **Attention heads**

 Most models don’t touch the attention heads much at all with a few minor exceptions. In this section, we'll introduce some variants of attention mechanism:

- **GQA / MQA** : Saving inference costs by reducing the number of heads
- **Sparse or sliding window attention** (used by GPT-4/Mistral) restricting the attention pattern to reduce compute cost.

##### **GQA/MQA** 

Let’s think about the compute involved for attention:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528202931619.png" alt="image-20250528202931619" style="zoom:50%;" />

- **Total arithmetric operations** ($𝑏𝑛𝑑^2$), **total memory accesses** ($𝑏𝑛𝑑 + 𝑏ℎ𝑛^2 + 𝑑^2$)
- Arithmetic intensity is high $𝑂((\frac{1}{n} + \frac{1}{bn})^{-1})$ that we can keep our GPUs running .

What about the *incremental* case when we generate text? The Key difference is that it can’t parallelize the generation process which means we needs to be step by step.

In this case, we need to incrementaly re-compute/update attention via the **KV cache**. [[This article](https://medium.com/@joaolages/kv-caching-explained-276520203249)] gives a nice explanation on KV cache.

<img src="https://miro.medium.com/v2/resize:fit:1400/1*uyuyOW1VBqmF5Gtv225XHQ.gif" alt="img" style="zoom:67%;" />

With KV cache, the **total arithmetric operations** is  ($𝑏𝑛𝑑^2$), **total memory accesses** ($𝑏𝑛^2𝑑 + 𝑛𝑑^2$). However, the Arithmetic intensity($O((\frac{n}{d} + \frac{1}{d})^{-1}$)  is not good, you need:

-  large batches ($b$)
- short seq length ($n$) or big model dimensions ($d$)

> The n/d term is difficult to reduce

Can we solve it?--use **MQA**(Multi Query Attention)! The key idea is that it has multiple queries, but just one dimension for keys and values.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528222713126.png" alt="image-20250528222713126" style="zoom:50%;" />

From the figure, we can see that we have much fewer items to move in and out of memory (KV Cache) .

- **Total memory access** ($𝑏𝑛𝑑 + 𝑏𝑛^2𝑘 + 𝑛𝑑^2$) 
- **Arithmetic intensity**: $O((\frac{1}{d} + \frac{n}{dh} + \frac{1}{b})^{-1})$

A recent extension is $$GQA$$(Group Query Attention) which Don’t go all the way to one dimension of KV – have fewer dims:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528223238544.png" alt="image-20250528223238544" style="zoom:50%;" />

It's a simple knob to control expressiveness (key-query ratio) and inference efficiency.

However, there is no free lunch. From [[Ainslie 2023](https://arxiv.org/pdf/2305.13245)], MQA/GQA sometimes hurt performance:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528223712292.png" alt="image-20250528223712292" style="zoom:50%;" />

##### **Sparse / sliding window attention**

Attending to the entire context can be expensive (quadratic). From [[Child et al 2019](https://arxiv.org/pdf/1904.10509)], we can build sparse / structured attention that trades off expressiveness vs runtime:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528223844364.png" alt="image-20250528223844364" style="zoom:50%;" />

Another variation on this idea is sliding window attention. It just use the main part of the strided pattern and let depth extend effective context. [[Mistral](https://arxiv.org/pdf/2310.06825)] use sliding window attention to train it.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528224428380.png" alt="image-20250528224428380" style="zoom:67%;" />

##### interleave ‘full’ and ‘LR’ attention

From [Cohere Command A](https://cohere.com/research/papers/command-a-technical-report.pdf), there is a standard trick--interleave ‘full’ and ‘LR’ attention which make every 4th layer is a full attention.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250528224720590.png" alt="image-20250528224720590" style="zoom: 67%;" />

- Long-range info via NoPE, short-range info via RoPE + SWA.

## MOE

In this section, we'll introduce **MOE(Mixture-Of-Expert)**, an architecture used by almost all the LLM:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530092553911.png" alt="image-20250530092553911" style="zoom:33%;" />

### The Basics 

What's the MOE? I think it's a confusing name. In fact, MOE just **replace big feedforward with (many) big feedforward networks and a selector layer**. And surprisingly you can increase the # experts without affecting FLOPs. The archticture is showed as follows:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530093333539.png" alt="image-20250530093333539" style="zoom:50%;" />

The next question is why so many LLM use MOE?

-  **Peformance**: [[Fedus et al 2022](https://arxiv.org/pdf/2209.01667)] shows that if two models have same FLOPs, more param does better. MOE can archive it compared to the dense model.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530093658110.png" alt="image-20250530093658110" style="zoom:40%;" />

- **Efficiency**: [[OLMOE](https://arxiv.org/pdf/2409.02060)] shows that it's faster to train MOE compared to dense model.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530094042230.png" alt="image-20250530094042230" style="zoom:40%;" />

- **Parallelism Consideration**: MOE can be parallelizable to many devices.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530094319797.png" alt="image-20250530094319797" style="zoom:50%;" />

From [[deepseek moe](https://arxiv.org/pdf/2401.06066)], they do some good recent ablation work on MoEs showing they’re generally good.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530095033957.png" alt="image-20250530095033957" style="zoom:50%;" />

However, there is no free lunch. [[Fedus et al 2022](https://arxiv.org/pdf/2209.01667)] confirm that if you want training MOE-based model, infrastructure is complex and  it only take advantages on multi node:

![image-20250530095523333](https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530095523333.png)

And [[Zoph et al 2022](https://arxiv.org/pdf/2202.08906)] confirm that the training objectives are somewhat heuristic (and sometimes unstable).

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530095722340.png" alt="image-20250530095722340" style="zoom:40%;" />

We'll take a closer look at MOE in the following sections.

### Routing function

You may think the router maybe some neural network architecture. Actually Many of the routing algorithms boil down to **choose top k**.From [[Fedus et al 2022](https://arxiv.org/pdf/2209.01667)], there are three types of router function:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530130226400.png" alt="image-20250530130226400" style="zoom:50%;" />

Almost all the MoEs do a standard **token choose topk** routing.  [[OLMOE](https://arxiv.org/pdf/2409.02060)] does some ablations to show shat **token choose topk** is overweight **expert choose topk** in performance.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530130812868.png" alt="image-20250530130812868" style="zoom:50%;" />

There are some common routing variants in detail:

- **Topk**: used in most MoEs(Switch Transformer (k=1) Gshard (k=2), Grok (2), Mixtral (2), Qwen (4), DBRX (4), DeepSeek (7))

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530131111148.png" alt="image-20250530131111148" style="zoom:60%;" />

- **Hashing** used as baseline 

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530131217695.png" alt="image-20250530131217695" style="zoom:50%;" />

- **RL to learn routes**: Used in some of the earliest work, not common now

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530131539000.png" alt="image-20250530131539000" style="zoom:50%;" />

- **Solve a matching problem**: Linear assignment for routing Used in various papers like [[Clark ‘22](https://proceedings.mlr.press/v162/clark22a/clark22a.pdf)]

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530132038165.png" alt="image-20250530132038165" style="zoom:50%;" />

The choose top-k router can be defined as follows:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530132240875.png" alt="image-20250530132240875" style="zoom: 33%;" />

### MOE Archticture Varaints

There are some variants of MOE architecture. [[Deepseek MOE](https://arxiv.org/pdf/2401.06066)] shows the evolution of the MOE architecture:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530161537178.png" alt="image-20250530161537178" style="zoom:40%;" />

- **Fine-Grained Expert Segmentation**: 

  - **Intuition**: In scenarios where the number of experts is limited, tokens assigned to a particular expert will be more likely to cover diverse types of knowledge. As a consequence, the designated expert will intend to learn vastly different types of knowledge in its parameters, and **they are hard to be simultaneously utilized**. However, if each token can be routed to more experts, diverse knowledge will gain the potential to be decomposed and learned in different experts respectively. In this context, each expert can still retain a high level of expert specialization, contributing to a more focused knowledge distribution across experts.
  - **Method**:  Segment the experts with a finer grain. The finer expert segmentation enables a more flexible and adaptable combination of activated experts. We segment each expert FFN into 𝑚 smaller experts by reducing the FFN intermediate hidden dimension to $\frac{1}{m}$ times its original size. The formula is:

  <img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530162816301.png" alt="image-20250530162816301" style="zoom:33%;" />

  - **Shared Expert Isolation**: 

    - **Intuition**: With a conventional routing strategy, tokens assigned to different experts may necessitate some common knowledge or information. As a result, multiple experts may converge in acquiring shared knowledge in their respective parameters, thereby resulting in redundancy in expert parameters. 
    - **Method**: , We further isolate $K_s$ experts to serve as shared experts. Regardless of the router module, each token will be deterministically assigned to these shared experts. In order to maintain a constant computational cost, the number of activated experts among the other routed experts will be decreased by  $K_s$

    <img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530162837966.png" alt="image-20250530162837966" style="zoom:33%;" />

    

From [[Deepseek MOE](https://arxiv.org/pdf/2401.06066)] ablations, they show that both **Fine-Grained Expert Segmentation** and **Shared Expert Isolation** contribute to stronger overall performance.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530163143239.png" alt="image-20250530163143239" style="zoom: 33%;" />

Finally, let's see the expert routing setups for recent MoEs:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530163332703.png" alt="image-20250530163332703" style="zoom:33%;" />

### Training objectives

In this section, we'll answer the hardest problem--how do we train MoEs. The major challenge is that we need sparsity for training-time efficiency. But sparse gating decisions are not differentiable.

There are several solutions:

- Reinforcment learning to optimize gating policies 
- Stochastic perturbations 
- Heuristic ‘balancing’ losses.

#### Reinforcment learning to optimize gating policies

From [[Clark et al 2022](https://proceedings.mlr.press/v162/clark22a/clark22a.pdf)], we can see that RL via REINFORCE does work, but not so much better that it’s a clear win:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530165436684.png" alt="image-20250530165436684" style="zoom:40%;" />

RL is the ‘right solution’ but gradient variances and complexity means it’s not widely used.

#### Stochastic approximations

It was introduced by [[Shazeer et al 2017](https://arxiv.org/pdf/1701.06538)] which make routing decisions are stochastic with gaussian perturbations. The computation formula is:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530170340549.png" alt="image-20250530170340549" style="zoom:33%;" />

The advantages of this method are:

- This naturally leads to experts that are a bit more robust. 
- The softmax means that the model learns how to rank K experts

Another approach like [[Fedus et al 2022](https://arxiv.org/pdf/2209.01667)] using stochastic jitter to improve the stability..It does a uniform multiplicative perturbation for the same goal of getting less brittle experts.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530171210309.png" alt="image-20250530171210309" style="zoom:30%;" />

However, this was later removed in [[Zoph et al 2022](https://arxiv.org/abs/2202.08906)] because they find it's useless:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530233942251.png" alt="image-20250530233942251" style="zoom:50%;" />

#### Heuristic ‘balancing’ losses

Almost all the model training MOE using heuristic balancing loss. The key issue when training MOE is that it always choose single MOE finally. However, systems efficiency requires that we use experts evenly. **Heuristic balancing loss** is introduced by [[Fedus et al 2022](https://arxiv.org/pdf/2209.01667)] to address the issue.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250530234328161.png" alt="image-20250530234328161" style="zoom:40%;" />

The derivative with respect to $p_i(x)$ is $\frac{\alpha N}{T^2}1\{argmaxp(x)=i\}$, so more frequent use = stronger downweighting .

Let's see example from deepseek v1 to v3:

- **Per-expert balancing** for v1: same as the switch transformer

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531000005436.png" alt="image-20250531000005436" style="zoom:50%;" />

- **Per-device balancing** for v2: the objective above, but aggregated by device

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531000041368.png" alt="image-20250531000041368" style="zoom:50%;" />

- **per-expert biases** for v3: Set up a per-expert bias (making it more likely to get tokens) and use online learning

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531000159783.png" alt="image-20250531000159783" style="zoom:50%;" />

Let's see some ablations from [[OLMOE](https://arxiv.org/abs/2409.02060)] for balancing loss:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531000255716.png" alt="image-20250531000255716" style="zoom:70%;" />

#### Training MoEs – the systems side

MoEs can parallelize nicely which means each FFN can fit in a device:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531000548144.png" alt="image-20250531000548144" style="zoom:80%;" />

And MOE parallelism enables additional kinds of parallelism:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531000710221.png" alt="image-20250531000710221" style="zoom:50%;" />

However, MoE routing allows for parallelism, but also some complexities. So [[MegaBlocks](https://arxiv.org/pdf/2211.15841)] uses smarter sparse MMs.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531144542464.png" alt="image-20250531144542464" style="zoom:30%;" />

#### Issues to train MOE

##### **Stochasticity of MoE Models**

There was speculation that LLM’s stochasticity was due to MoE.Why would a MoE have additional randomness? From  [[MegaBlocks](https://arxiv.org/pdf/2211.15841)], we can see that token dropping from routing happens at a *batch* level and this means thatother people’s queries can drop your token!

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531145404143.png" alt="image-20250531145404143" style="zoom:33%;" />

##### Stability

From [[Zoph 2022](https://arxiv.org/pdf/2202.08906)], sparse models often suffer from training instabilities worse than those observed in standard densely-activated Transformers due to softmax computation.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531145932625.png" alt="image-20250531145932625" style="zoom:50%;" />

![image-20250531145951754](https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531145951754.png)

The solution is that Use Float 32 just for the expert router and sometimes with an aux z-loss.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531150319798.png" alt="image-20250531150319798" style="zoom:50%;" />

[[OLMOE](https://arxiv.org/pdf/2409.02060)] shows that using z-loss for MoE router training is more stable.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531150633294.png" alt="image-20250531150633294" style="zoom:60%;" />

##### **Fine-Tuning**

Often sparse MoEs can overfit on smaller fine-tuning data.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531150749492.png" alt="image-20250531150749492" style="zoom:80%;" />

- [[Zoph 2022](https://arxiv.org/pdf/2202.08906)] solution: finetune non-MoE MLPs

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531150939329.png" alt="image-20250531150939329" style="zoom:50%;" />

- DeepSeek solution: use lots of data 1.4M SFT

![image-20250531151031289](https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531151031289.png)

#### **Other training methods - upcycling**

**Upcycling** means using a pre-trained LM to initialize a MoE.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531151358378.png" alt="image-20250531151358378" style="zoom:70%;" />

For example, [[MiniCPM](https://arxiv.org/pdf/2404.06395)] use upcycling to train MoE version of model. It shows that upcycling can help gain performance.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531151755284.png" alt="image-20250531151755284" style="zoom:100%;" />

[[**Qwen MoE**](https://arxiv.org/pdf/2412.15115)] initialized from the Qwen 1.8B model top-k=4, 60 experts w/ 4 shared. 

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531152002449.png" alt="image-20250531152002449" style="zoom:40%;" />

### DeepSeek v1-v2-v3

Finally, we’ll walk through the DeepSeek MoE architecture.

- **[DeepSeek V1](https://arxiv.org/pdf/2401.02954) (16B – 2.8 active):** 

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531152252804.png" alt="image-20250531152252804" style="zoom:50%;" />

- **[DeepSeek V2](https://arxiv.org/abs/2405.04434) (236B – 21 active):** 

![image-20250531152524705](https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531152524705.png)

- **[DeepSeek V3](https://arxiv.org/pdf/2412.19437) (671B – 37 active):** 

![image-20250531152643478](https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531152643478.png)

For [DeepSeek V3](https://arxiv.org/pdf/2412.19437), they also introduce the MLA(Multihead, latent attention) to help gain efficiency:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531152804040.png" alt="image-20250531152804040" style="zoom:67%;" />

The **basic idea** is to express the Q, K, V as functions of a lower-dim, ‘latent’ activation $c_t$.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531152908840.png" alt="image-20250531152908840" style="zoom:50%;" />

The **benefits** is that when KV-caching, we only need to store $c^{KV}_t$, which can be much smaller. However, MLA also introduce some **complexity** which means rope conflicts with MLA-style caching:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531153250908.png" alt="image-20250531153250908" style="zoom:67%;" />

The solution is to have a few non-latent key dimensions that can be rotated.

For [DeepSeek V3](https://arxiv.org/pdf/2412.19437), they also introduce the **MTP**(Multi-Token Prediction) which has small, lightweight models that predict multiple steps ahead.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531153508685.png" alt="image-20250531153508685" style="zoom:47%;" />

The computation formula is:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250531153544914.png" alt="image-20250531153544914" style="zoom:43%;" />
