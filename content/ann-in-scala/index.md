+++
title="Artificial Neural Network in Scala"
date=2021-02-04
draft = true

[extra]
category="blog"

[taxonomies]
tags = ["deep learning", "machine learning", "gradient descent"]
categories = ["scala"]
+++

_Deep Learning_ is a part of machine learning methods which are based on artificial neural networks. Some of the deep learning architectures are deep neural networks.
Deep neural network is an artificial neural network (ANN) with multiple layers between the input and output layers. There are different types of neural networks, but they always have
neurons, synapses, weights, biases and functions.

[Scala](https://www.scala-lang.org/) is a full-stack multi-paradigm programming language which is famous for its innovations in JVM eco-system. Scala is also popular language thanks to Apache Spark, Kafka and Flink projects which are implemented in it. 

# Scope 

In this aritcle we are going to implement ANN from scrtach in Scala. This implementation will consist of:

1. Mini-library for sub-set of Tensor calculus
1. Mini-library for data preparation
1. A DSL for Neural Network creation including layers, neurons, weights
1. Plugable weights optimizer
1. Plugable backpropogation algorithm based on mini-batch gradient descent
1. Plugable implementation of activation and loss functions
1. Plugable training metric calculation

Everything will be implemented in pure Scala from scratch without using any third-party code. When I say plugable I mean extendable, i.e. a user
can provide own implementation by implementing Scala trait (type classes).

Before juming into the code, I will guide you through the basic calculus, which you need to know to implement neural network tranining and optimization algorithms.
Those calculus we need to know is liniear algebra and a little bit of differential calculus,

# ANN Jargon

I assume you are a bit familiar with machine learning or even deep learning. Nevertheless,bBelow table will be useful to match 
deep learning terminology with further Scala implementation. Some of the variable names in Scala code will be directly based 
on the deep learning name definitions, so that it is important to know why some variable is named as `z` and another one as `w`.


| Code Symbol / Type    | Description   | Designed in Code as |
| ------------- |:-------------:|:-------------:|
| x             | input data for each neuron | 2-dimensional tensor, i.e. matrix |
| y , actual            | target data we know in advance in training set | 1-d tensor, i.e. vector |
| yHat , predicted      | output of the neural network during the training or single prediction | 1-d tensor |
| w      | layer weight, a.k.a model paramaters | 2-d tensor |
| b | layer bias, part of the layer parameters | 1-d tensor
|z | layer activation, calculated as <br/> `x * w + b` | 2-d tensor |
| f, actFunc | activation function to activate neuron. Specific implementation: sigmoid, relu | Scala function |
|a | layer activity, calculated as f(z)| 2-d tensor |
| Neuron | keep sstate of a neuron (x, z, a) | case class |
| Weight | keeps state of a weight at particular training cycle, epoch| case class |
| Layer | keeps layer configuration: number of neurons, activation function for all neurons in the layer | case class |
| error | it is result of yHat - y (predicted - actual) | 1-d tensor |
| lossFunc | loss/cost function to calculate a value of incorrect predictions. Specific implementation: mean squared error | Scala function |
| epochs | number of iterations to train ANN | integer > 0 |
| accuracy | % of correct predictions on train or test data sets | as double number between 0 and 1 |
| learningRate | numeric paramater used in weights update | as double number, usually 0.01 or 0.001 | 

# Dataset

I have taken a Churn Modeling dataset which predicts whether a customer is going to leave the bank or not. 
This dataset is traveling accross many tutorials at the Internet, 
so that you can find a couple different code implementations among dozens of articales using the same data, so do I.

```csv
RowNumber,CustomerId,Surname,CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited
1,15634602,Hargrave,619,France,Female,42,2,0,1,1,1,101348.88,1
2,15647311,Hill,608,Spain,Female,41,1,83807.86,1,0,1,112542.58,0
3,15619304,Onio,502,France,Female,42,8,159660.8,3,1,0,113931.57,1
4,15701354,Boni,699,France,Female,39,1,0,2,0,0,93826.63,0
```

The last column is `y`, i.e. the value we want our ANN to predict between 0 and 1. Rest of the columns are `x` data also known as features.
We won't use columns for `x`, but drop some them.

# Linear Algebra

Basic idea of neural network implementation is to leverage math principals for maxtrix and vector multiplication. As neural network is going to be fully conected from layer to layer, we can represent learning and opmization algorithms as following:

## Dimensions

### One training example 

It is one data record:

1. X: 12 input columns as 1 x 12 matrix
1. W1: 12 x 6 matrix + 6 x 1 matrix for baises
1. W2: 6 x 6 matrix + 6 x 1 matrix for baises
1. W3: 6 x 1 matrix + 1 x 1 matrix for baises
1. Y hat: scalar number

Number of hidden layers and neurons are parameters to tune by machine learning engineer. So we set 2 hidden layers with 6 neurons each. Our last layer
is single neuron that produces the prediction.

### Multiple training examples

Mini-batch approach, i.e. multiple training examples at once. Let's take 16 as batch size:

1. X: 16 x 12 matrix
1. W1: 12 x 6 matrix + 6 x 1 matrix for baises
1. W2: 6 x 6 matrix + 6 x 1 matrix for baises
1. W3: 6 x 1 matrix + 1 x 1 matrix for baises
1. Y hat: 16 x 1 matrix

As you can see, our matrices are equal in rows at input and output layers.
That means we can input any number of rows through the neural network at once when we do training or prediction.
 
## Single Batch Trace

In order to see what is going in with the state of neural network, let's feed one signle batch to it.

### X - input

Below matrix is our X, which is input data from the training or test set. The values it contains is a result of 
data preparation step, which I will explain in detials further:

```bash
sizes: 16x12, Tensor2D[Float]:
[[-0.032920357,1.1368092,-0.579135,-0.6694619,-0.9335201,0.3765018,-1.0295835,-1.0824665,-0.9476086,0.7312724,0.8284169,-0.05904571]
 [-0.12171035,-0.86240697,-0.579135,1.464448,-0.9335201,0.26486462,-1.3750359,0.2695743,-0.9476086,-1.340666,0.8284169,0.1443363]
 [-0.97732306,1.1368092,-0.579135,-0.6694619,-0.9335201,0.3765018,1.0431306,1.4932814,2.0728939,0.7312724,-1.1834527,0.16957337]
 [0.6128251,1.1368092,-0.579135,-0.6694619,-0.9335201,0.041590314,-1.3750359,-1.0824665,0.5626426,-1.340666,-1.1834527,-0.19572]
 [1.8316696,-0.86240697,-0.579135,1.464448,-0.9335201,0.48813897,-1.0295835,0.94235253,-0.9476086,0.7312724,0.8284169,-0.463582]
 [0.17694691,-0.86240697,-0.579135,1.464448,1.0502101,0.59977615,1.0431306,0.7527129,0.5626426,0.7312724,-1.1834527,0.82049227]
 [1.6056587,1.1368092,-0.579135,-0.6694619,1.0502101,1.2695991,0.69767827,-1.0824665,0.5626426,0.7312724,0.8284169,-1.7176533]
 [-1.9943721,-0.86240697,1.6928561,-0.6694619,-0.9335201,-1.0747813,-0.33867878,0.7735395,3.583145,0.7312724,-1.1834527,0.267966]
 [-0.9853949,1.1368092,-0.579135,-0.6694619,1.0502101,0.59977615,-0.33867878,1.20919,0.5626426,-1.340666,0.8284169,-0.5388685]
 [0.49174783,1.1368092,-0.579135,-0.6694619,1.0502101,-1.2980556,-1.0295835,1.0890474,-0.9476086,0.7312724,0.8284169,-0.5972788]
 [-0.7674558,1.1368092,-0.579135,-0.6694619,1.0502101,-0.85150695,0.35222593,0.563331,0.5626426,-1.340666,-1.1834527,-0.44364995]
 [-1.0176822,-0.86240697,-0.579135,1.464448,1.0502101,-1.6329671,-0.68413115,-1.0824665,0.5626426,0.7312724,-1.1834527,-0.5125319]
 [-1.1871903,1.1368092,-0.579135,-0.6694619,-0.9335201,-0.5165955,1.7340354,-1.0824665,0.5626426,0.7312724,-1.1834527,-1.4233431]
 [-0.5979476,1.1368092,-0.579135,-0.6694619,-0.9335201,-1.52133,0.0067735757,-1.0824665,0.5626426,-1.340666,-1.1834527,1.5672718]
 [0.09622873,-0.86240697,-0.579135,1.464448,-0.9335201,-0.40495834,0.69767827,-1.0824665,0.5626426,0.7312724,0.8284169,-0.70219]
 [-0.05713581,-0.86240697,1.6928561,-0.6694619,1.0502101,0.71141326,-0.68413115,1.2265866,0.5626426,-1.340666,0.8284169,-0.731704]]
```

### W - 1st hidden layer

weight:

```bash
sizes: 12x6, Tensor2D[Float]:
[[0.0031250487,0.6230519,0.41888618,0.89568454,0.6927525,0.7887961]
 [0.30450296,0.75655645,0.075620584,0.898596,0.66731954,0.06079619]
 [0.21693613,0.89243406,0.6251479,0.080811165,0.30963784,0.87972105]
 [0.006676915,0.05886997,0.88085854,0.29817313,0.19820364,0.6823392]
 [0.73550576,0.49408147,0.99867696,0.71354216,0.9676805,0.09009225]
 [0.19121544,0.021707054,0.53959745,0.74587476,0.16132912,0.08185377]
 [0.2528674,0.562563,0.17039675,0.7291027,0.41844574,0.4336123]
 [0.8275197,0.5867702,0.1692482,0.102723576,0.8936942,0.12275006]
 [0.15337862,0.55374163,0.7993138,0.73106086,0.29611018,0.6279454]
 [0.15933406,0.5840742,0.42520604,0.44090283,0.13000321,0.25581995]
 [0.8168607,0.25407365,0.6668799,0.277898,0.13848923,0.94559854]
 [0.61593276,0.8569094,0.83978665,0.12022303,0.097834654,0.9559516]]
```

bias:

```bash
sizes: 6, Tensor1D[Float]:
[0.0,0.0,0.0,0.0,0.0,0.0]
```

### W - 2st hidden layer

weight:

```bash
sizes: 6x6, Tensor2D[Float]:
[[0.77492374,0.20006068,0.8301353,0.8226056,0.726444,0.54590976]
 [0.8728169,0.83197665,0.5453676,0.7730933,0.77980715,0.20573096]
 [0.8222075,0.94630164,0.29234344,0.7667057,0.3600455,0.26467463]
 [0.5196553,0.3935514,0.23351222,0.18136671,0.01824836,0.25099826]
 [0.8864608,0.64109814,0.3031471,0.18872173,0.5463185,0.26470202]
 [0.102771536,0.92541504,0.21454614,0.8614344,0.10369446,0.76455885]]
```

bias:

```bash
sizes: 6, Tensor1D[Float]:
[0.0,0.0,0.0,0.0,0.0,0.0]
```

### W - 3st hidden layer

weight:

```bash
sizes: 6x1, Tensor2D[Float]:
[[0.63033116]
 [0.6078242]
 [0.022346135]
 [0.62451136]
 [0.89858407]
 [0.5960952]]
```

bias:

```bash
sizes: 1, Tensor1D[Float]:
[0.0]
```

### Y Hat - Output layer

Predictions per each data sample out of 16 samples:

```bash
sizes: 16, Tensor1D[Float]:
[0.8810671,0.49800232,0.86774874,0.49800232,0.99999976,0.83703655,0.49800232,0.49800232,0.9976035,0.49800232,0.9999994,0.49800232,0.8031284,0.49800232,0.9775441,0.86094683]
```

__________________________________

Above trace is just for the first traning batch at specific epoch. Do not try to make some sense out these digits in weights, biases and outputs.
These matrices are going to change their values a lot after running traning loop 100 times (epocs) with N batches each.