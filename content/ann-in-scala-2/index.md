+++
title="Artificial Neural Network in Scala - part 2"
date=2021-02-05
draft = false

[extra]
category="blog"

[taxonomies]
tags = ["deep learning", "machine learning", "gradient descent"]
categories = ["scala"]
+++

In this aritcle we are going to implement ANN from scrtach in Scala. It is contuniation of [the first part](../ann-in-scala-1)

This implementation will consist of:

1. Mini-library for sub-set of Tensor calculus
1. Mini-library for data preparation
1. A DSL for Neural Network creation including layers, neurons, weights
1. Plugable weights optimizer
1. Plugable backpropogation algorithm based on mini-batch gradient descent
1. Plugable implementation of activation and loss functions
1. Plugable training metric calculation

Everything will be implemented in pure Scala without using any third-party code. 
By plugable I mean extendable, i.e. a user can provide own implementation by implementing Scala trait (type classes).


TODO...................