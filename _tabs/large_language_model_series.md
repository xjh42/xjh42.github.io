---
title: Language Model From Strach
layout: post
icon: fas fa-star
order: 1
---

Language models serve as the cornerstone of modern natural language processing (NLP) applications and open up a new paradigm of having a single general purpose system address a range of downstream tasks. As the field of artificial intelligence (AI), machine learning (ML), and NLP continues to grow, possessing a deep understanding of language models becomes essential for scientists and engineers alike. This blog series will provide readers with a comprehensive understanding of language models by walking them through the entire process of developing their own.

Language models are all about efficiency. Given resources(**data + hardware (compute, memory, communication bandwidth)**), how do you train the best model? 

> For example, given a Common Crawl dump and 32 H100s for 2 weeks, what should you do?

The series is organized as follows:

![image-20250524172044968](https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250524172044968.png)

## Basics

The goal in basics module is to introduce the basic version of the full pipeline working. It contains following components:(**tokenization, model architecture, training**)

### Tokenization

Tokenizers convert between strings and sequences of integers (tokens)

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250524172403684.png" alt="image-20250524172403684" style="zoom:50%;" />

The intuition of tokenization is to break up string into popular segments. We'll introduce the most popular tokenizer: Byte-[Pair Encoding (BPE) tokenizer](https://arxiv.org/abs/1508.07909).

However, there are also tokenizer-free approaches: [**ByT5**](https://arxiv.org/abs/2105.13626), [**MEGABYTE**](https://arxiv.org/pdf/2305.07185), [**BLT**](https://arxiv.org/abs/2412.09871), [**T-FREE**](https://arxiv.org/abs/2406.19223). They use bytes directly, promising, but have not yet been scaled up to the frontier model.


### Architecture

Almost all language models is transformer-based model which introduce by [[Vaswani+ 2017\]](https://arxiv.org/pdf/1706.03762.pdf).  However, in language model, we often use decoder-only transformer, its architecture is shown as follows:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250524173427185.png" alt="image-20250524173427185" style="zoom:50%;" />

There are lots of variants:

- **Activation function variants**: ReLU, SwiGLU [[Shazeer 2020\]](https://arxiv.org/pdf/2002.05202.pdf)
- **Positional encoding variants**: sinusoidal, RoPE [[Su+ 2021\]](https://arxiv.org/pdf/2104.09864.pdf)
- **Normalization varaints**: LayerNorm, RMSNorm [[Ba+ 2016\]](https://arxiv.org/pdf/1607.06450.pdf)[[Zhang+ 2019\]](https://arxiv.org/abs/1910.07467)

- **Placement variants of normalization**: pre-norm versus post-norm [[Xiong+ 2020\]](https://arxiv.org/pdf/2002.04745.pdf)
- **MLP variants**: dense, mixture of experts [[Shazeer+ 2017\]](https://arxiv.org/pdf/1701.06538.pdf)
- **Attention varaints**: full, sliding window, linear [[Jiang+ 2023\]](https://arxiv.org/pdf/2310.06825.pdf)[[Katharopoulos+ 2020\]](https://arxiv.org/abs/2006.16236)
- **Lower-dimensional attention varaints**: group-query attention (GQA), multi-head latent attention (MLA) [[Ainslie+ 2023\]](https://arxiv.org/pdf/2305.13245.pdf)[[DeepSeek-AI+ 2024\]](https://arxiv.org/abs/2405.04434)
- **State-space models**: Hyena [[Poli+ 2023\]](https://arxiv.org/abs/2302.10866)

### Training

Training module approximately contains following parts:

- **Optimizer**:  (e.g., AdamW, Muon, SOAP) [[Kingma+ 2014\]](https://arxiv.org/pdf/1412.6980.pdf)[[Loshchilov+ 2017\]](https://arxiv.org/pdf/1711.05101.pdf)[[Keller 2024\]](https://kellerjordan.github.io/posts/muon/)[[Vyas+ 2024\]](https://arxiv.org/abs/2409.11321)
- **Learning rate schedule** (e.g., cosine, WSD) [[Loshchilov+ 2016\]](https://arxiv.org/pdf/1608.03983.pdf)[[Hu+ 2024\]](https://arxiv.org/pdf/2404.06395.pdf)
- **Batch size** (e..g, critical batch size) [[McCandlish+ 2018\]](https://arxiv.org/pdf/1812.06162.pdf)

- **Regularization** (e.g., dropout, weight decay)

- **Hyperparameters** (number of heads, hidden dimension): grid search

### Detailed Posts
You can find more details in following posts:

- **[Language Models 1: Tokenization](https://xjh42.github.io/posts/LLM-Tokenizer/)**
- **[Language Models 2: Pytorch](https://xjh42.github.io/posts/LLM-Pytorch/)**
- **[Language Models 3: The Evolution of Transformer-based Language Models](% link _posts/2025-05-31-LLM-The-Evolution-of-Transformer-based-Language-Models.md %})**

## System

In system module, our goal is to  squeeze the most out of the hardware. 

### Kernels

A GPU (A100) looks like:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250524192453902.png" alt="image-20250524192453902" style="zoom:50%;" />

GPU is device in a computer, so it need fetch data from memory in order to do compution. We can make an analogy:

warehouse-->DRAM, factory -->SRAM. The bandwidth cost is matters.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250524192707414.png" alt="image-20250524192707414" style="zoom:50%;" />

A very important discipline is that we need to <u>organize computation to maximize utilization of GPUs by minimizing data movement</u>.

We can  write kernels in CUDA/Triton/CUTLASS/ThunderKittens.

### Parallelism

Nowadays we can use multiple GPUs to speed up computation. The following picture shows a cluster of 8 A100 GPUs.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250524193818902.png" alt="image-20250524193818902" style="zoom:50%;" />

Data movement between GPUs is even slower, but same **minimize data movement** principle holds. 

- We can use **collective operations** (e.g., gather, reduce, all-reduce)
- **Shard** (parameters, activations, gradients, optimizer states) across GPUs

In order to split computation ,we can do  {data,tensor,pipeline,sequence} parallelism.

### Inference

Inference means **generate tokens given a prompt by using model**.

Inference is also needed for reinforcement learning, test-time compute, evaluation. Globally, inference compute (every use) exceeds training compute (one-time cost). Often it contains two phases: prefill and decode.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250524215223360.png" alt="image-20250524215223360" style="zoom:50%;" />

- **Prefill** (similar to training): tokens are given, can process all at once (compute-bound)

- **Decode**: need to generate one token at a time (memory-bound)

There are several methods to speed up decoding:

- **Use cheaper model** (via model pruning, quantization, distillation)
- **Speculative decoding**: use a cheaper "draft" model to generate multiple tokens, then use the full model to score in parallel (exact decoding!)
- **Systems optimizations**: KV caching, batching

## Scaling Law

The goal in scaling law module is to <u>do experiments at small scale, predict hyperparameters/loss at large scale</u>. Given a FLOPs budget ($C$), use a bigger model ($N$) or train on more tokens ($D$)?

Often we use compute-optimal scaling laws:  [[Kaplan+ 2020\]](https://arxiv.org/pdf/2001.08361.pdf)[[Hoffmann+ 2022\]](https://arxiv.org/pdf/2203.15556.pdf).

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250524220708000.png" alt="image-20250524220708000" style="zoom: 50%;" />

An common equation that we use is:
$$
D^* = 20 N^*
$$


> For example, 1.4B parameter model should be trained on 28B tokens

## Data

In data part, we need answer what capabilities do we want the model to have. Multilingual? Code? Math? Different capabilities need different type of data. The following picture shows different types of data.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250524222834621.png" alt="image-20250524222834621" style="zoom: 33%;" />

### Evaluation

Evaluation means given a model how do you evaluate it's good or not. Here are some methods:

- **Perplexity**: textbook evaluation for language models
- **Standardized testing** (e.g., MMLU, HellaSwag, GSM8K)
- **Instruction following** (e.g., AlpacaEval, IFEval, WildBench)
- **Scaling test-time compute**: chain-of-thought, ensembling

- **LM-as-a-judge**: evaluate generative tasks
- **Evaluate full system**: RAG, agents

### Data curation

Data does not just fall from the sky.  We need collect them from different source. The sources contains webpages crawled from the Internet, books, arXiv papers, GitHub code, etc. Often we need consider following things:

- fair use to train on copyright data [[Henderson+ 2023\]](https://arxiv.org/pdf/2303.15715.pdf)
- license data [[article\]](https://www.reuters.com/technology/reddit-ai-content-licensing-deal-with-google-sources-say-2024-02-22/)

The format of data contains **HTML, PDF, directories**.

### Data processing

Data processing often contains three parts:

- **Transformation**: convert HTML/PDF to text (preserve content, some structure, rewriting
- **Filtering**: keep high quality data, remove harmful content (via classifiers)
- **Deduplication**: save compute, avoid memorization; use Bloom filters or MinHash

## Aligment

So far, a **base model** is raw potential, very good at completing the next token. Alignment makes the model actually useful.

The goals of alignment are:

- Get the language model to follow instructions
- Tune the style (format, length, tone, etc.)
- Incorporate safety (e.g., refusals to answer harmful questions)

### Supervised finetuning (SFT)

Given Instruction data: (prompt, response) pairs(Data often involves human annotation):

```python
sft_data: list[ChatExample] = [
        ChatExample(
            turns=[
                Turn(role="system", content="You are a helpful assistant."),
                Turn(role="user", content="What is 1 + 1?"),
                Turn(role="assistant", content="The answer is 2."),
            ],
        ),
    ]
```

The intuition of SFT is that base model already has the skills, just need few examples to surface them.[[Zhou+ 2023\]](https://arxiv.org/pdf/2305.11206.pdf). The supervised learning method is to  fine-tune model to maximize $ p(response | prompt) $.

### Learning from feedback

Now we have a preliminary instruction following model. Let's make it better without expensive annotation. We will introduce learning from feedback method.

- **Preference data**:  We generate multiple responses using model (e.g., [A, B]) to a given prompt to get the preference data.

  We then provides preferences (e.g., A < B or A > B).

  ```python
  preference_data: list[PreferenceExample] = [
          PreferenceExample(
              history=[
                  Turn(role="system", content="You are a helpful assistant."),
                  Turn(role="user", content="What is the best way to train a language model?"),
              ],
              response_a="You should use a large dataset and train for a long time.",
              response_b="You should use a small dataset and train for a short time.",
              chosen="a",
          )
      ]
  ```

- **Verifiers**:  we can use formal verifiers (e.g., for code, math) or learned verifiers(train against an LM-as-a-judge).

- **The algorithm**:

**Proximal Policy Optimization (PPO)** from reinforcement learning [[Schulman+ 2017\]](https://arxiv.org/pdf/1707.06347.pdf)[[Ouyang+ 2022\]](https://arxiv.org/pdf/2203.02155.pdf)

**Direct Policy Optimization (DPO)**: for preference data, simpler [[Rafailov+ 2023\]](https://arxiv.org/pdf/2305.18290.pdf)

**Group Relative Preference Optimization (GRPO)**: remove value function [[Shao+ 2024\]](https://arxiv.org/pdf/2402.03300.pdf)

## Efficiency drives design decisions

Today, we are compute-constrained, so design decisions will reflect squeezing the most out of given hardware.

- **Data processing**: avoid wasting precious compute updating on bad / irrelevant data    

- **Tokenization**: working with raw bytes is elegant, but compute-inefficient with today's model architectures.

- **Model architecture**: many changes motivated by reducing memory or FLOPs (e.g., sharing KV caches, sliding window attention)

- **Training**: we can get away with a single epoch!

- **Scaling law**s: use less compute on smaller models to do hyperparameter tuning

- **Alignment**: if tune model more to desired use cases, require smaller base models
