---
title: "Backpropagation and Multivariable Calculus"
description: >-
  A quick intro on backpropagation and multivariable calculus for deep learning
date: 2025-04-20 12:01:00 +0000
categories:
  - AI
  - Gradients for Backpropagation
tags:
  - ai
  - deep learning
  - maths
  - backpropagation
  - tensor calculus
  - index notation
pin: false
math: true
---

In this post, I'll introduce the core of deep learning framework: **automatic differentiation**. In the first part, I'll introduce the thoery for automatic differentiation--mostly about **reverse automatic differentiation**. In the second part, I'll introduce how to implement a basic deep learning framework based on autodiff.

# Theory

So why calculating differentation is important in deep learning? Let's first look at the three basic elements in deep learning:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@main/uPic/image-20250210093510526.png" alt="image-20250210093510526" style="zoom: 50%;" />

We can see computing the loss function gradient with respect to hypothesis class parameters is the most common operation in machine learning.
