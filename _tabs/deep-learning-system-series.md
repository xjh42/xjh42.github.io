title: Deep Learning System Series
layout: post
icon: fas fa-stream
order: 1
---

Here is a collection of posts explaining how deep learning system works. 

Deep learning methods have revolutionized a number of fields in Artificial Intelligence and Machine Learning in recent years. The widespread adoption of deep learning methods have in no small part been driven by the widespread availability of easy-to-use deep learning systems, such as PyTorch and TensorFlow. But despite their widespread availability and use, it is much less common for students to get involved with the internals of these libraries, to understand how they function at a fundamental level. But understanding these libraries deeply will help you make better use of their functionality, and enable you to develop or extend these libraries when needed to fit your own custom use cases in deep learning.

The goal of this blog series is to provide students an understanding and overview of the “full stack” of deep learning systems, ranging from the high-level modeling design of modern deep learning systems, to the basic implementation of automatic differentiation tools, to the underlying device-level implementation of efficient algorithms.

## Blog Series
Included in the blog series:

* [DLSys: ML Refresher / Softmax Regression]({% link _posts/2025-02-01-DLSys-ML-Refresher-Softmax-Regression.md %}) gives a refresher of the basics of machine learning.
* [DLSys: Manual Neural Networks]({% link _posts/2025-02-24-DLSys-Manual-Neural-Networks.md %}) explains how a simple nerual network works and how backpropagation works.
* [DLSys: Automatic Differentiation]({% link _posts/2025-02-07-DLSys-Automatic-Differentiation.md %}) is the core algorithm which makes it easier to calculate gradients than backpropagation.
* [DLSys: Nerual Network Library Abstraction]({% link _posts/2025-03-01-DLSys-Nerual-Network-Library-Abstraction.md %}) explains how deep learning framework like pytorch works.
* [DLSys: Hardware Acceleration]({% link _posts/2025-03-13-DLSys-Hardware-Acceleration.md %}) explains how to use morden hardware like GPU to accelate you neural network computing!.
* [DLSys: Convolutional Networks]({% link _posts/2024-03-20-DLSys-Convolutional-Networks.md %}) explains why you need CNN instead of MLP.
* [DLSys: Sequence Modeling and Recurrent Networks]({% link _posts/2024-03-2-DLSys-Sequence-Modeling-and-Recurrent-Networks.md %}) explains why you need RNN for language modeling tasks.
* [DLSys: Transformers and Attention']({% link _posts/2025-04-01-DLSys-Transformers-and-Autoregressive-Models.md %}) explains transformer which is very popular now.

## Project Implementation

Furthermore, you can use these blog series to implement a simple but power pytorch-like framework which we can it **needle**. You can find my implementation here: [needle implementation](https://github.com/xjh42/needle_implementation)
