---
title: 'DLSys: Sequence Modeling and Recurrent Networks'
date: 2024-03-24 17:22:54
tags:
- rnn
categories:
- dlsys

pin: false
math: true
---

## Theory

### Sequence Modeling

For the previous posts, we make prediction assuming input and output pairs $(x^{(i)}, y^{(i)})$  is **independent identically distributed(i.i.d)**.It means the previous result donnot affect current result. In pratice, many cases where **the input/output pairs are given in a specific sequence**, and we need to use the information about this sequence to help us make predictions.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240824173117956.png" alt="image-20240824173117956" style="zoom:67%;" />

There are several typical examples of sequence modeling:

- **Part of speech tagging**: Given a sequence of words, determine the part of speech of each word.**A word’s part of speech depends on the context in which it is being used**, not just on the word itself.

![image-20240824173242176](https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240824173242176.png)

- **speech to text**: Given a audio signal (assume we even know the word boundaries, and map each segment to a fix-sized vector descriptor), determine the corresponding transcription. Again, context of the words is extremely important. Because many words' pronunciation are same.  (see e.g., any bad speech recognition system that attempts to “wreck a nice beach”)

![image-20240824173452738](https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240824173452738.png)

- **autoregressive prediction**: A special case of sequential prediction where the elements to predict is the next element in the sequence.Common e.g., in time series forecasting, language modeling, and other use cases. We strongly rely on the context of the sentance to predict the next word.

![image-20240824173627222](https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240824173627222.png)

### Recurrent Neural Networks

Recurrent neural networks (RNNs) is a model to save the sequence model problem.  RNN maintain a **hidden state** over time, which is a function of the current input and previous hidden state. The previous hidden state contains the context of the previous inputs. Therefore, hidden state use the current input and a list of previous inputs to make a prediction. 

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240826092058694.png" alt="image-20240826092058694" style="zoom:50%;" />
$$
h_t = f(W_{hh}h_{t-1} + W_{hx}x_t + b_h)
$$

$$
y_t = g(W_{yh}h_t + b_y)
$$

Where $f$ and $g$ are activation function. $W_{hh}$ , $W_{hx}$, $W_{yh}$ are weights, and $b_y$, $b_h$ are bias term. And $x \in R^n$,  $y \in R^{k}$, $h_t \in R^d$, $W_{hh} \in R^{d \times d}$, $W_{yh} \in R^{k \times d}$, $W_{hx} \in R^{d \times n}$, $b_h \in R^d$, $b_y \in R^k$.

After we define the RNN model, the next question is how to train RNN? Given a sequence of inputs and target outputs$(x_1, ..., x_T, y^{*}_1, ..., y^{*}_T)$, we can train an RNN using backpropagation through time, which just involves “unrolling” the RNN over the length of the sequence, then relying mostly on **autodiff**. Without autodiff, we cannot solve the problem, because we cannot write the gradient of the rnn model.

```python
opt = Optimizer(params = (W_hh, W_hx, W_yh, b_h, b_y))
h[0] = 0
l = 0
for t = 1,...,T:
  h[t] = f(W_hh * h[t-1] + W_hx * x[t] + b_h)
  y[t] = g(W_yh * h[t] + b_y)
  l += Loss(y[t], y_star[t])
l.backward()
opt.step()
```



As you can see, the challenge for training RNNs is similar to that of training deep MLP networks, becasuse the sequence maybe long and the rnn is complicated.

- **Exploding activations/gradients**: Because we train RNNs on long sequences, if the weights/activation of the RNN are scaled poorly, the hidden activations (and therefore also the gradients) will grow unboundedly with sequence length. For example, we use below initialization, the gradient will soon be NaN which cannot be stored in the 32-bit floating number.

  ![image-20240826094116833](https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240826094116833.png)

- **Vanishing activation/gradients**: Similarly, if weights are too small then information from the inputs will quickly decay with time (and it is precisely the “long range” dependencies that we would often like to model with sequence models). So the context of the previous  inputs will decay.

  ![image-20240826094344394](https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240826094344394.png)

To solve **Exploding activations/gradients** problem, we can use other activation functions. ReLU is a bad activation function because it can grow unboundedly. We can use sigmod and tanh activation function.

![image-20240826094802233](https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240826094802233.png)

But the problem **Vanishing activation/gradients** still be unsolved. Creating large enough weights to not cause activations/gradients to vanish requires being in **the “saturating” regions of the activations**, where gradients are very small ⟹ still have vanishing gradients

How solve this problems? Use LSTM!

### LSTMs

Long short term memory (LSTM) cells are a particular form of hidden unit update that avoids (some of) the problems of vanilla LSTMs. It make two changes to avoid vanishing activation/gradients.

- Step 1:  Divide the hidden unit into two components, called (confusingly) the **hidden state** and the **cell state**

  ![image-20240826095305637](https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240826095305637.png)

  - Step 2: Use a very specific formula to update the hidden state and cell state (throwing in some other names, like “forget gate”, “input gate”, “output gate” for good measure)

    ![image-20240826095513339](https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240826095513339.png)

    where $i_t \in R^d$, $f_t \in R^d$,$g_t \in R^d$, $o_t \in R^d$, $W_{hh} \in R^{4d \times d}$, $h_t \in R^d$ , $W_{hx} \in R^{4d \times n}$

    

Why LSTM works? The factor of $f_t$ and $i_t$ can control the context information. Close to 0 --> not mantain the context, Close 1 --> context information will be untoched.

![image-20240826100216569](https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240826100216569.png)

### Beyond "simple" sequential Models

We'll introduce a list of aplication of RNN.

- **Seq2Seq model**: To give you a short glimpse of the kind of things you can do with RNNs/LSTMs beyond “simple” sequence prediction, consider the task of **trying to translate between languages**. 

  Can concatenate two RNNs together, one that “only” processes the sequence to create a final hidden state (i.e., no loss function, encoder); then a section that takes in this initial hidden state, and “only” generates a sequence(decoder). Why this model works? Because the translation task is not a one-one mapping problem.

  ![image-20240826100555240](https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240826100555240.png)

  $h_5$ contains the summary of the context.

- **Bidirectional RNNs**: RNNs can use only the sequence information up until time $t$ to predict $y_t$.This is sometimes desirable (e.g., autoregressive models). But sometime undesirable (e.g., language translation where we want to use “whole” input sequence)

  Bi-directional RNNs stack a forwardrunning RNN with a backward-running RNN: information from the entire sequence to propagates to the hidden state. So we can use the full context to predict!

  ![image-20240826101002495](https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20240826101002495.png)

## Implementing RNNs

Codelab notebook links: [implementing RNNs](https://colab.research.google.com/drive/1834vcXfVcwne_h1RwVyz_oNeLtbk8dRz?usp=sharing)

