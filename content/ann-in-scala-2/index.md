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
By plugable I mean extendable, i.e. a user can provide own implementation by implementing Scala trait.

Neural network and data preprosseng APIs are inspired by Keras and scikit-learn libraries.

# Tensor Library

Before starting our journey into the world of linear algebra we need good support for Tensor calculus such as
mutliplication, addition, substraction, transponding operarions. Without these operations, we will clutter the
main algotithm so that another person, who will be reading our code, is going to be lost itself. It is very easy to be blown
away by pile of code which is trying to mimic math. Scala is perfect language to implement math expression as
it supports custom operands by using symbols as methods, i.e. we can implement "*" or any other math operations for
our custom type `Tensor`.

Below we define a trait Tensor for generic type T. Later we set boundaries for T. It must have `given` 
instances of ClassTag and Numeric types.

```scala
sealed trait Tensor[T]:
  type A
  def data: A
  def length: Int
  def sizes: List[Int]
  def cols: Int
  
extension [T: ClassTag: Numeric, U: ClassTag](t: Tensor[T])
    // dot product    
    def *(that: Tensor[T]): Tensor[T] = Tensor.mul(t, that)
    def map(f: T => U): Tensor[U] = Tensor.map[T, U](t, f)
    def -(that: T): Tensor[T] = Tensor.subtract(t, Tensor0D(that))
    def -(that: Tensor[T]): Tensor[T] = Tensor.subtract(t, that)
    def +(that: Tensor[T]): Tensor[T] = Tensor.plus(t, that)    
    def sum: T = Tensor.sum(t)        
    def split(fraction: Float): (Tensor[T], Tensor[T]) = 
        Tensor.split(fraction, t)
    // Hadamard product
    def multiply(that: Tensor[T]): Tensor[T] = Tensor.multiply(t, that)
    def batches(
        batchSize: Int
    ): Iterator[Array[Array[T]]] = Tensor.batches(t, batchSize)    
```

In extension section we add lots of operations that our generic Tensor is going to support. Some of them are symbolic like
`*` and `-`. Other are operations are more traditional methods such as `map` or `sum`. 
Note that `*` and `multiply` are two different operations. From math perspective, the first one is a dot product
another one is a Hadamard product. Most of the time, we will use dot product operation, 
however in one place Hadamard product is going to be used (backpropagation part).

Before checking some of the implementations for Tensor operations, let's look on 3 cases of Tensor itself.

```scala
case class Tensor0D[T: ClassTag](data: T) extends Tensor[T]:
  type A = T
  ....

case class Tensor1D[T: ClassTag](data: Array[T]) extends Tensor[T]:
  type A = Array[T]
  ....

case class Tensor2D[T: ClassTag](data: Array[Array[T]]) extends Tensor[T]:
  type A = Array[Array[T]]  
  ....
```

Look at how A type is set based on the Tensor dimension.

From math perspective, first instance is scalar number, second is vector and third is matrix. Of course, we could implement
tensors in more generic way and invent some N-dimensional array that would support 3, 4 and any number of dimenions,
but I think from learning perspective, making more concrete hard-coded classes would be easier to understand the whole ANN 
implementation.

## Matmul

Let's look only at one important operation from Tensor API which is dot product.

```scala
def mul[T: ClassTag: Numeric](a: Tensor[T], b: Tensor[T]): Tensor[T] =
  (a, b) match
    case (Tensor0D(data), t) =>
      scalarMul(t, data)
    case (t, Tensor0D(data)) =>
      scalarMul(t, data)
    case (Tensor1D(data), Tensor2D(data2)) =>
      Tensor2D(matMul(asColumn(data), data2))
    case (Tensor2D(data), Tensor1D(data2)) =>
      Tensor2D(matMul(data, asColumn(data2)))
    case (Tensor1D(data), Tensor1D(data2)) =>
      Tensor1D(matMul(asColumn(data), Array(data2)).head)
    case (Tensor2D(data), Tensor2D(data2)) =>
      Tensor2D(matMul(data, data2))

  private def matMul[T: ClassTag: Numeric](
      a: Array[Array[T]],
      b: Array[Array[T]]
  ): Array[Array[T]] =
    assert(
      a.head.length == b.length,
      "The number of columns in the first matrix should be equal " + 
      s"to the number of rows in the second, ${a.head.length} != ${b.length}"
    )
    val rows = a.length
    val cols = b.headOption.map(_.length).getOrElse(0)
    val res = Array.ofDim[T](rows, cols)

    for i <- (0 until rows).indices do
      for j <- (0 until cols).indices do
        var sum = summon[Numeric[T]].zero
        for k <- b.indices do
          sum = sum + (a(i)(k) * b(k)(j))
        res(i)(j) = sum
    res    
```

1. First we select specific type of multiplication based on the tensor dimension. 
1. If tensor is not scalar, then we try to use matrix multiplication. Here, if some of the operands is vector we make that vector as matrix
with one column according to math convension. 
1. Later we check operands dimensions, as they must obey rules of
matrix multiplication. If rules are not met we throw an error. No Scala `Either` type or other error modeling is used to not clutter the code. Our goal is to stay as close as possible to math and keep balance between using types and readbility.

# Neural Network DSL

Let's define a trait for abstract model that can learn:

```scala
sealed trait Model[T]:
  def train(x: Tensor[T], y: Tensor[T], epochs: Int): Model[T]
  def predict(x: Tensor[T]): Tensor[T]
  def reset(): Model[T]
  def currentWeights: List[Weight[T]]
  def losses: List[T]
```

The first two methods are the main ones. 

1. We can use `train` to provide input features as `x` and target values as `y`, specify
number of training cycles as `epochs` to learn the right parameters for future predictions.
1. `predict` allows us to infer target value by giving only `x` data
1. `reset` cleans model's weights, so that it initializes again upon next training
1. `currentWeights` and `losses` are giving weights and losses of the last training cycle.

Model is a stateful thing. It keeps list of parameteres called weights and biases of type `List[Weight[T]]`.
These parameters are the heart of the model. They are mutating on every training epoch and data batch.

## Model initialization

Before designing neural network traning API, let's look at entities we need:

```scala
sealed trait ActivationFunc[T] extends (Tensor[T] => Tensor[T]):
  def apply(x: Tensor[T]): Tensor[T]
  def derivative(x: Tensor[T]): Tensor[T]

sealed trait Loss[T]:
  def apply(actual: Tensor[T], predicted: Tensor[T]): T

sealed trait Layer[T]:
  def units: Int
  def f: ActivationFunc[T]

case class Dense[T](f: ActivationFunc[T], units: Int = 1) extends Layer[T]

case class Weight[T](
  w: Tensor[T], b: Tensor[T], 
  f: ActivationFunc[T], units: Int
)

/*
 * z - before activation = w * x
 * a - activation value
 */
case class Activation[T](x: Tensor[T], z: Tensor[T], a: Tensor[T])
```

We have modeled network parameters as traits with implementations as case classes. Later we use 
them to create an instance of the model.

```scala
trait RandomGen[T]:
  def gen: T

case class Sequential[T: ClassTag: RandomGen: Numeric, U: Optimizer](
    lossFunc: Loss[T],
    learningRate: T,
    metric: Metric[T],
    batchSize: Int = 16,
    weightStack: Int => List[Weight[T]] = (_: Int) => List.empty[Weight[T]],    
    weights: List[Weight[T]] = Nil,
    losses: List[T] = Nil
) extends Model[T]:

  def add(layer: Layer[T]): Sequential[T, U] =
    copy(weightStack = (inputs) => {
      val currentWeights = weightStack(inputs)
      val prevInput =
        currentWeights.reverse.headOption.map(_.units).getOrElse(inputs)
      val w = random2D(prevInput, layer.units)
      val b = zeros(layer.units)
      (currentWeights :+ Weight(w, b, layer.f, layer.units))
    })
```

There are bunch of parameters that we need in simple sequential model with fully connected layers:

1. Generic type `T` is numeric type of the data which can be `Float`, `Double` or `Int`. Most of the time you need numbers with floating point.
1. Random generator can be provided as contextual abstraction (given instance). It is used to initialize
weights and biases for every layer.
1. Generic `U` is a type of opmization algorithm that we use in backpropagation part of the training cycle. Also given as type class instance.
1. `learningRate` and `batchSize` are hyper-parameters to be tuned externally.
1. `weightStack` is a function that construct list of initial layers based on the provided earlier Layer configuration via
method `add`. `weightStack` is called by `train` method to create inital list of weights, if weights are empty. If they 
are not empty, they are reused.

This is how user suppose to use such API:

```scala
val accuracy = accuracyMetric[Float]
  
val ann = Sequential[Float, SimpleGD](
  binaryCrossEntropy,
  learningRate = 0.05f,
  metric = accuracy,
  batchSize = 32
)
  .add(Dense(relu, 6))
  .add(Dense(relu, 6))    
  .add(Dense(sigmoid))
```

There is a type `SimpleGD` that picks up a required instance of `Optimizer` implementation. See details below.

## Training loop

`train` method runs `trainEpoch` multiple times, which is equal to `epochs` parameter. 
Every training epoch returns new weights list, which is used again for the next epoch. This loop may run, for example, 100 times.
Also, we collect a list of average loss values and print a metric value. We have set `accuracy` metric as per code earlier.

```scala
def train(x: Tensor[T], y: Tensor[T], epochs: Int): Model[T] =
  lazy val inputs = x.cols
  lazy val actualBatches = y.batches(batchSize).toArray
  lazy val xBatches = x.batches(batchSize).zip(actualBatches).toArray
  lazy val w = getWeights(inputs)

  val (updatedWeights, epochLosses) =
    (1 to epochs).foldLeft((w, List.empty[T])) {
      case ((weights, losses), epoch) =>
        val (w, avgLoss, metricValue) = trainEpoch(xBatches, weights)
        val metricAvg = metric.average(x.length, metricValue)
        println(
          s"epoch: $epoch/$epochs, avg. loss: $avgLoss, 
          ${metric.name}: $metricAvg"
        )
        (w, losses :+ avgLoss)
    }
  copy(weights = updatedWeights, losses = epochLosses)
```

`trainEpoch` is implementing forward and backpropagation for every data sample batch:

```scala
private def trainEpoch(
    batches: Array[(Array[Array[T]], Array[Array[T]])],
    weights: List[Weight[T]]
) =
  val (w, l, metricValue) =
    batches.foldLeft(weights, List.empty[T], 0) {
      case ((weights, batchLoss, metricAcc), (xBatch, yBatch)) =>
        // forward
        val activations = activate(xBatch.as2D, weights)
        val actual = yBatch.as2D          
        val predicted = activations.last.a          
        val error = predicted - actual          
        val loss = lossFunc(actual, predicted)

        // backward
        val updated = summon[Optimizer[U]].updateWeights(
          weights,
          activations,
          error,
          learningRate
        )
        val metricValue = metric.calculate(actual, predicted)
        (updated, batchLoss :+ loss, metricAcc + metricValue)
    }    
  (w, getAvgLoss(l), metricValue)
```

### Gradient Descent Optimizer

Now let's look at optimizer code. It implements gradient descent algorithm:

```scala
sealed trait Optimizer[U]:
  def updateWeights[T: Numeric: ClassTag](
    weights: List[Weight[T]],
    activations: List[Activation[T]],
    error: Tensor[T],
    learningRate: T
  ): List[Weight[T]]

type SimpleGD
```

In order to update weights optimizer needs:
1. the list of weights itself
1. current activations for all layers
1. calculated error: yHat vs. y
1. learningRate parameter, which is static for the entire traning cycle

Earlier, we used a static type `SimpleGD` when we constructed `Sequential` model. In future, we can extend optimizers with other algorithms.

Data batching is happening outside of the optimizer, in the `train` method.
We can select either full batch or mini-batch training by specifing a number of record in the batch. 

Actual implementation of the gradient descent optimization:

```scala
given Optimizer[SimpleGD] with
  override def updateWeights[T: Numeric: ClassTag](
      weights: List[Weight[T]],
      activations: List[Activation[T]],
      error: Tensor[T],
      learningRate: T
  ): List[Weight[T]] =      
    weights
      .zip(activations)
      .foldRight(
        List.empty[Weight[T]],
        error,
        None: Option[Tensor[T]]
      ) {
        case (
              (Weight(w, b, f, u), Activation(x, z, _)),
              (ws, prevDelta, prevWeight)
            ) =>            
          val delta = (prevWeight match {
            case Some(pw) => prevDelta * pw.T
            case None     => prevDelta
          }) multiply f.derivative(z)

          val partialDerivative = x.T * delta
          val newWeight = w - (learningRate * partialDerivative)
          val newBias = b - (learningRate * delta.sum)
          val updated = Weight(newWeight, newBias, f, u) +: ws
          (updated, delta, Some(w))
      }
      ._1
```

Weights update starts from tail and moves to the head of the list, i.e. from the last layer to the first hidden layer.
`weights` and `activations` are equal in lenght, since the last one is produced via the weight list during the forward propagation. 

The complex part is calculating the `delta` that we use for partial derivative. 

1. Initial `delta` is equal to `error`. Next layer is calculating `delta` on its own, which is a dot product of previous layer `delta` and weights.
1. Last layer does not have previous weights.
1. Every `delta` is then multiplied by activation function derivative `f.derivative(z)`.
1. The rest part is simpler and more or less linear. We calculate `partialDerivative` and update layer's weight and bias tensors.
1. We pass current layer weight and delta to the next layer. Usage of `foldRight` helps us easily to pass these parameters to next layer.

This folding loop returns updated list of weights, which is of course equal in list lenght to the original list length.

# Data Preparation

Before we start learning, we need to prepare initial data for the training. 
Unfortunatelly, data preparation requires us quite a lot of code to write. 

```scala
def createEncoders[T: Numeric: ClassTag](
    data: Tensor2D[String]
  ): Tensor2D[String] => Tensor2D[T] =
  val encoder = LabelEncoder[String]().fit(data.col(2))
  val hotEncoder = OneHotEncoder[String, T]().fit(data.col(1))
  
  val label = x => encoder.transform(x, 2)
  val hot = x => hotEncoder.transform(x, 1)
  val typeTransform = (x: Tensor2D[String]) => transform[T](x.data)
  
  label andThen hot andThen typeTransform
///////////////////////////////////////////////////////

val dataLoader = TextLoader(Path.of("data", "Churn_Modelling.csv")).load()
val data = dataLoader.cols[String](3, -1)

val encoders = createEncoders[Float](data)
val numericData = encoders(data)
val scaler = StandardScaler[Float]().fit(numericData)

val prepareData = (d: Tensor2D[String]) => {
  val numericData = encoders(d)
  scaler.transform(numericData)
}

val x = prepareData(data)
val y = dataLoader.cols[Float](-1)
```

1. First, we load raw data from a CSV file, then we select all columns between 3-rd and last one.
1. Initial data is String type, later we choose Numeric data type.
1. We compose label, one-hot encoders and type transformers into a function inside the `createEncoders` function. That allows us to use 
`prepareData` function later for validation dataset.
1. As `y` we take the last column of the dataset.

I am not going to describe the entire code of data preparation classes. The goal of encoders is to 
prepare data for deep neural network to learn by normalizing all columns a per their individual means and 
standard devitation. Also, we encode categorical colums using 0 and 1 via one-hot encoding approach.

# Training Run

```scala
val ((xTrain, xTest), (yTrain, yTest)) = (x, y).split(0.2f)
val model = ann.train(xTrain, yTrain, epochs = 100)
```

Then we split `x` and `y` data into two 80% and 20% parts for training and testing accordingly.
And finally execute the training for 100 `epochs`.

```bash
sbt:ann> run
[info] running starter
epoch: 1/100, avg. loss: 0.30220446, accuracy: 0.782
epoch: 2/100, avg. loss: 0.30736533, accuracy: 0.811375
epoch: 3/100, avg. loss: 0.30326372, accuracy: 0.818125
epoch: 4/100, avg. loss: 0.30306807, accuracy: 0.818875
epoch: 5/100, avg. loss: 0.3028989, accuracy: 0.820125
epoch: 6/100, avg. loss: 0.30242646, accuracy: 0.82025
epoch: 7/100, avg. loss: 0.3018655, accuracy: 0.8195
epoch: 8/100, avg. loss: 0.30157945, accuracy: 0.81975
epoch: 9/100, avg. loss: 0.3014126, accuracy: 0.819875
epoch: 10/100, avg. loss: 0.30122074, accuracy: 0.819625
epoch: 11/100, avg. loss: 0.3009277, accuracy: 0.81975
epoch: 12/100, avg. loss: 0.30088165, accuracy: 0.82
epoch: 13/100, avg. loss: 0.30078012, accuracy: 0.8205
epoch: 14/100, avg. loss: 0.30074772, accuracy: 0.8205
epoch: 15/100, avg. loss: 0.30070674, accuracy: 0.8205
epoch: 16/100, avg. loss: 0.30053124, accuracy: 0.82025
epoch: 17/100, avg. loss: 0.2976923, accuracy: 0.81975
epoch: 18/100, avg. loss: 0.2536276, accuracy: 0.84275
epoch: 19/100, avg. loss: 0.24473017, accuracy: 0.85675
epoch: 20/100, avg. loss: 0.24557488, accuracy: 0.857125
epoch: 21/100, avg. loss: 0.24528943, accuracy: 0.857875
epoch: 22/100, avg. loss: 0.2451054, accuracy: 0.857125
epoch: 23/100, avg. loss: 0.24494325, accuracy: 0.857375
epoch: 24/100, avg. loss: 0.24466132, accuracy: 0.857
epoch: 25/100, avg. loss: 0.24451153, accuracy: 0.857625
epoch: 26/100, avg. loss: 0.24442412, accuracy: 0.857375
epoch: 27/100, avg. loss: 0.24431105, accuracy: 0.857625
epoch: 28/100, avg. loss: 0.24418788, accuracy: 0.857375
epoch: 29/100, avg. loss: 0.2440211, accuracy: 0.85775
epoch: 30/100, avg. loss: 0.24400905, accuracy: 0.85725
epoch: 31/100, avg. loss: 0.24397133, accuracy: 0.85725
epoch: 32/100, avg. loss: 0.24386458, accuracy: 0.857375
epoch: 33/100, avg. loss: 0.24389265, accuracy: 0.8575
epoch: 34/100, avg. loss: 0.24378827, accuracy: 0.857375
epoch: 35/100, avg. loss: 0.24381112, accuracy: 0.857875
epoch: 36/100, avg. loss: 0.2437651, accuracy: 0.857875
epoch: 37/100, avg. loss: 0.24369456, accuracy: 0.85775
epoch: 38/100, avg. loss: 0.24377964, accuracy: 0.857625
epoch: 39/100, avg. loss: 0.2435442, accuracy: 0.85775
epoch: 40/100, avg. loss: 0.24363366, accuracy: 0.858125
epoch: 41/100, avg. loss: 0.24358764, accuracy: 0.858125
epoch: 42/100, avg. loss: 0.24355079, accuracy: 0.858375
epoch: 43/100, avg. loss: 0.24369176, accuracy: 0.858125
epoch: 44/100, avg. loss: 0.24361038, accuracy: 0.858125
epoch: 45/100, avg. loss: 0.24359651, accuracy: 0.858125
epoch: 46/100, avg. loss: 0.24361634, accuracy: 0.858125
epoch: 47/100, avg. loss: 0.24357627, accuracy: 0.858
epoch: 48/100, avg. loss: 0.24334462, accuracy: 0.85825
epoch: 49/100, avg. loss: 0.24335352, accuracy: 0.858
epoch: 50/100, avg. loss: 0.24341401, accuracy: 0.858375
epoch: 51/100, avg. loss: 0.24324806, accuracy: 0.8585
epoch: 52/100, avg. loss: 0.24296027, accuracy: 0.858
epoch: 53/100, avg. loss: 0.24271448, accuracy: 0.85775
epoch: 54/100, avg. loss: 0.24256946, accuracy: 0.85825
epoch: 55/100, avg. loss: 0.24257207, accuracy: 0.858125
epoch: 56/100, avg. loss: 0.24284393, accuracy: 0.8585
epoch: 57/100, avg. loss: 0.2430726, accuracy: 0.85825
epoch: 58/100, avg. loss: 0.2431463, accuracy: 0.858125
epoch: 59/100, avg. loss: 0.24277006, accuracy: 0.857625
epoch: 60/100, avg. loss: 0.2423336, accuracy: 0.8585
epoch: 61/100, avg. loss: 0.24251764, accuracy: 0.858125
epoch: 62/100, avg. loss: 0.24255769, accuracy: 0.858625
epoch: 63/100, avg. loss: 0.2427412, accuracy: 0.85825
epoch: 64/100, avg. loss: 0.2428449, accuracy: 0.85825
epoch: 65/100, avg. loss: 0.24228723, accuracy: 0.858625
epoch: 66/100, avg. loss: 0.24231568, accuracy: 0.85875
epoch: 67/100, avg. loss: 0.24237442, accuracy: 0.858125
epoch: 68/100, avg. loss: 0.24238351, accuracy: 0.8585
epoch: 69/100, avg. loss: 0.24219948, accuracy: 0.859125
epoch: 70/100, avg. loss: 0.24231845, accuracy: 0.858875
epoch: 71/100, avg. loss: 0.24243066, accuracy: 0.85875
epoch: 72/100, avg. loss: 0.2423754, accuracy: 0.859
epoch: 73/100, avg. loss: 0.24225388, accuracy: 0.8585
epoch: 74/100, avg. loss: 0.2420498, accuracy: 0.858875
epoch: 75/100, avg. loss: 0.24199313, accuracy: 0.858625
epoch: 76/100, avg. loss: 0.2420193, accuracy: 0.858875
epoch: 77/100, avg. loss: 0.24175513, accuracy: 0.85875
epoch: 78/100, avg. loss: 0.24191435, accuracy: 0.859625
epoch: 79/100, avg. loss: 0.2418117, accuracy: 0.85925
epoch: 80/100, avg. loss: 0.24193105, accuracy: 0.859125
epoch: 81/100, avg. loss: 0.24175763, accuracy: 0.859375
epoch: 82/100, avg. loss: 0.24183328, accuracy: 0.859375
epoch: 83/100, avg. loss: 0.24171984, accuracy: 0.85975
epoch: 84/100, avg. loss: 0.2419013, accuracy: 0.859125
epoch: 85/100, avg. loss: 0.24182202, accuracy: 0.859625
epoch: 86/100, avg. loss: 0.24179217, accuracy: 0.859875
epoch: 87/100, avg. loss: 0.2416485, accuracy: 0.859875
epoch: 88/100, avg. loss: 0.24175707, accuracy: 0.86
epoch: 89/100, avg. loss: 0.24161652, accuracy: 0.85975
epoch: 90/100, avg. loss: 0.24164297, accuracy: 0.8595
epoch: 91/100, avg. loss: 0.24179684, accuracy: 0.859625
epoch: 92/100, avg. loss: 0.2417788, accuracy: 0.859875
epoch: 93/100, avg. loss: 0.24164356, accuracy: 0.859875
epoch: 94/100, avg. loss: 0.24161309, accuracy: 0.8595
epoch: 95/100, avg. loss: 0.24144296, accuracy: 0.85975
epoch: 96/100, avg. loss: 0.24163213, accuracy: 0.85975
epoch: 97/100, avg. loss: 0.24149604, accuracy: 0.86
epoch: 98/100, avg. loss: 0.24137497, accuracy: 0.85925
epoch: 99/100, avg. loss: 0.24164252, accuracy: 0.859625
epoch: 100/100, avg. loss: 0.24153009, accuracy: 0.85975
training time: 5.654 in sec
```

We can see that `accuracy` is increasing quite quick. Also, loss value is becoming stable already after 20 epochs.

Entire training for 8000 data samples takes less than 6 seconds.

# Testing

## Single Test

```scala
// Single test
val example = TextLoader(
  "n/a,n/a,n/a,600,France,Male,40,3,60000,2,1,1,50000,n/a"
).cols[String](3, -1)
val testExample = prepareData(example)
val exited = 
  predictedToBinary(model.predict(testExample).as1D.data.head) == 1
println(s"Exited customer? $exited")
```

## Dataset Test

We have left 20% of the initial data for testing purposes. So now we can check trained model accuracy on new data 
that model had never seen:

```scala
val testPredicted = model.predict(xTest)
val value = accuracy(yTest, testPredicted)
println(s"test accuracy = $value")  
```

Model accuracy on unseen data is quite as well:

```bash
test accuracy = 0.8625
```

# Python Implementation

Almost the same implementation in Python takes much langer to train the model. 
Although, we are using a bit more advanced optimizer such as `Adam`.

```python
ann.compile(optimizer = 'adam', \
  loss = 'binary_crossentropy', metrics = ['accuracy'])

import time
start = time.process_time()
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
end = time.process_time() - start
print(f"training time = {end} sec")
```

```bash
training time = 24.495086
```

Entire code can be checked here: [tensorflow-ann-python](https://github.com/novakov-alexey/tensorflow-ann-python/blob/main/artificial_neural_network.py)

# Summary

We have seen that Scala implementation looks very conciese thanks to the great language designer.
It also works faster than Python implementation.