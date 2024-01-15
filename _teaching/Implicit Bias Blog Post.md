---
title: "Implicit Bias in Deep Learning"
collection: teaching
permalink: /ML/ImplicitBiasinDL
excerpt: 'A set of [notes](https://fekonrad.github.io/files/Implicit_Bias__Literature_Overview%20(1).pdf) attempting to summarize the literature on implicit bias of gradient descent.'
date: 2024-09-01
---

In many applications of modern Deep Learning, neural networks tend to be heavily overparametrized, meaning the amount of trainable parameters far exceeds the amount of training data. 
In such overparametrized regimes there tend to be many parameter configurations which lead to a network which perfectly fits (in the case of regression) or perfectly classifies (in the case of classification) the training data. However, not all functions that fit the data are qualitatively the same.

![Two Functions that fit the Data](/images/ImplicitBiasPic1.png)

In the picture above, both functions (blue and orange) fit the data (red points) exactly, but the blue curve clearly has a certain desirable "smoothness" that the yellow curve does not have. 

**Question: If we train a neural network by minimizing mean squared error without explicitly regularizing the network, does the fitted neural network tend to look more like the blue curve (smooth) or the orange curve (not smooth)?**

Surprisingly, It turns out that for wide overparametrized networks, gradient descent (and many of its variants) tend to result in surprisingly smooth / regularized functions. This is called the *implicit bias of gradient descent*, as the procedure of optimizing a neural network by gradient descent tends to be "biased" towards particular functions. 

It is an active field of research in ML Theory to establish results of this kind. In [these notes](https://fekonrad.github.io/files/Implicit_Bias__Literature_Overview%20(1).pdf)
  I try to provide an introduction to the field and summarize parts of this research area. 
This is still a work in progress and I will continue to update these notes in the future!

