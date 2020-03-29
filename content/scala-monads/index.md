+++
title="Monads in Scala"
date=2020-03-28
toc_enabled = false
draft = true

[extra]
category="blog"

[taxonomies]
tags = ["fp"]
categories = ["scala"]
+++
Once you start dig deeper into Scala and its suitability for functional programming, you meet Monads. In this blog post, we will explore Monads in Scala:
their usage and usefulness. 

{{ resize_image(path="scala-monads/flatmap-all-the-things.png", width=620, height=620, op="fit_width") }}

## What is Monad?

You have probably already heared this quote:

{% quote(author="Saunders Mac Lane") %}A monad is just a monoid in the category of endofunctors{% end %}

[More details on StackOverflow answer](https://stackoverflow.com/a/3870310/6176274)

Well, that does not bring much help. Obviously, Monad is not just Scala pattern, but it is something what is coming from Category Theory. 
However, we are not going to touch Category Theory in general, but let's say that Monad definition is coming from Abstract theory of Mathematics.

I like another defininition of Monad, which is given in the  book "Functional Programming in Scala":

{% quote(author="Chiusano, Bjarnason") %}Monad is an abstract interface{% end %}

It is more clear for programmers. Before we clarify in details what Monad is let's look at example of Mondas in Scala standard library.
This might already click for you that Monad is not something from aliens:

- Option
- Either
- List
- Future
- Map
- Set
- Stream
- Vector
- Try 

... and others

## What makes thing a Monad?

There are several minimum combinations of function which make some type a Monad. One of the popular minimum set is two functons:

- **flatMap** - also known as `bind`
- **unit** - also known as `pure` in [Cats library](https://typelevel.org/cats/typeclasses/monad.html#monad-instances) or `apply` in pure Scala

These two functions implemented for some type bring powerful abstraction to write complex programms easy.

## Make your Monad

Monad sometimes reminds a container to work with its values using special interface. If we model Monad ourselves, then it may look like box with a thing
inside, which we operate one using `flatMap` one more useful function `map`:

```scala
class Box[A](v: A) { 

  def flatMap[B](f: A => Box[B]): Box[B] = f(v)
  
  def map[B](f: A => B): Box[B] = flatMap(a => new Box(f(a)))
}
```

**map** - is implemented in terms of `flatMap` + `unit` (i.e. Box class constructor). So we can implement `map` for any kinf of Monads, as we will that see later.

Now let's use `Box ` Monad to show some usage example:

```scala
scala> new Box(1)
res2: Box[Int] = 1

scala> res2.map(_ + 1)
res3: Box[Int] = 2

scala> res3.flatMap(i => new Box(1 + i))
res5: Box[Int] = 3
```

`Box` contains single integer value and allows us to manipulate it without leaving `Box` context, i.e. our result is always a `Box[T]`.
We can also make varible `v` as public and read it when needed. `Box` behaves similarly to non-empty single element list.
It is hard to say when this particular `Box` Monad will be useful looking at above example. However, it should give you an idea how Monad implementation may look like.

## Scala Examples

**List**

List operates on collection of values. 

```scala
scala> val l = List(1,2,3) // <-- unit
l: List[Int] = List(1, 2, 3)

scala> l.map(_ + 1)
res0: List[Int] = List(2, 3, 4)

scala> l.flatMap(i => List(i + 1))
res1: List[Int] = List(2, 3, 4)
```

**Option**

Option has two sub-types: `Some` and `None`. 

`Some` is like non-empty single element list, similar to Box Monad example above.
`None` ignores application of lambda function in `flatMap` or `map`.

```scala
val isOn = Some(1) // <-- unit
val isBlack = None // <-- unit without any argument

def makeCoffee: Option[String] = Some(1)

scala> isOn
         .flatMap(_ => isBlack
         .flatMap(_ => makeCoffee))

res0: Option[String] = None
```

Example above won't return value of `isOn` variable because the first `flatMap` call returns `None` because of `isBlack`, so that second `flatMap` even won't be called.

## Generic Monad

We have already seen example of at least 3 Monads above. In order to dettach definition of Monad from its concrete implementation like
List or Option, let us define abstract Monad interface using Scala High-Order types feature:

```scala
  trait Monad[F[_]] extends Functor[F] {
    def unit[A](a: => A): F[A]

    def flatMap[A,B](ma: F[A])(f: A => F[B]): F[B]

    def map[A,B](ma: F[A])(f: A => B): F[B] = 
      flatMap(ma)(a => unit(f(a))) 
  }

  trait Functor[F[_]] {
    def map[A,B](fa: F[A])(f: A => B): F[B]
  }
```

Functor is one more astraction which is more simpler than Monad. It requires only `map` implementation. We can say that every Monad also a Functor. 
Functor is also coming from the Category Therory. I decided to mention it here, because you will freuently find it in the context of Monads,
when learning functional programming in general. Abstract Monad interface can also implement map in terms of `flatMap` and unit functions, 
so that `map` is implemented automatically for any concrete implementation of some Monad.

## Function application in flatMap

You have probably noticed that `f` function application in `flatMap` and `map` depends on the concrete Monad instance. In one case the lambda
function we pass to the `flatMap` is always executed, in another cases not. Examples:

"f" applied when:

- Option[A]: is Some(A)
- Either[A, B]: is Right(B)
- List[A]: is non-empty
- Future[A]: is ready

Even though `flatMap` behaves differently on concrete Monad instance, there is still great benefits to use them in any ordinary program.
In order to classify some type as a Monad, one needs to comply with **Monad Laws** and that is closing the definition of Monads. Let's look 
at Monad laws before we more further to practical examples.

## Monad Laws

### 1. Identity

Resulf of a function which creates Monad instance using `unit` is equal to application of this function over already created Monad instance.


Example:

```scala
def f(x: Int): Option[Int] = Some(x)

scala> Some(1).flatMap(f) == f(1)
res0: Boolean = true

scala> f(1) == Some(1).flatMap(f)
res1: Boolean = true
```

Abstract definition of Identity Law:

#### 1.1 Left identity

```scala
def f[A](x: A): Monad[A] = ???

flatMap(unit(x))(f) == f(x) 
```

#### 1.2 Right identity

```scala
f(x) == flatMap(unit(x))(f)
```

### 2. Associative

Application of `f1` and `f2` functions one by one yields the same result as applying them within the first `flatMap`.

Example:

```scala
def f1(a: Int): Option[Int] = Some(a + 1)
def f2(a: Int): Option[Int] = Some(a * 2)

scala> Some(1).flatMap(f1).flatMap(f2)
res0: Option[Int] = Some(4)

scala> Some(1).flatMap(a => f1(a).flatMap(f2))
res1: Option[Int] = Some(4)
```

Abstract definition:

```scala
def f1[A](a: A): Monad[A]
def f2[A](a: A): Monad[A]

if x is a Monad instance,

flatMap(flatMap(x)(f1))(f2) == flatMap(x)(a => flatMap(f1(a))(f2))
```

## Functor Laws

### 1. Identity

Example:

```scala
map(Some(1))(a => a) == Some(1)
```

Abstract definition:

```scala
map(x)(a => a) == x  // the same value returned
```

### 2. Associative

Example:

```scala
val f1 = (n: Int) => n + 1
val f2 = (n: Int) => n * 2

map(map(Some(1))(f1))(f2) // Some(4)
            == 
map(Some(1))(f2 compose f1) // Some(4)
```

Standard Scala function `compose` return a funtion whicj applies f1 and then f2 on the result of the first function.

Abstract definition:

```scala
map(map(x)(f1))(f2) == map(x)(f2 compose f1) 
```

## Application of Monads

Using Monads we can do sequential composition. If we have several values in form of Option, we can sequence them into logic program,
which evaluates next value based on the behaviour of `flatMap` of the previous value. 

### Compose Option

```scala
final case class Coffee(name: String)

val isOn = Some(1)
val coffeeName = Some("black")
val makeCoffee = (name: String) => Some(Coffee(name))

for {
  _ <- isOn
  name <- coffeeName
  coffee <- makeCoffee(name)
} yield coffee

scala> Option[Coffee] = Some(Coffee(black))
```

Final result of this program is Some(..) value. However, it could result into None, if one these three values is None.

### Compose Either

The following three functions return Either Monad, so that we can compose them into a sequence.

```scala
case class Cluster(pods: Int)

def validateNamespace(ns: String): Either[String, Unit] = Right(())
def clusterExists(ns: String): Either[Cluster, Unit] = Right(())
def createCluster(ns: String, cluster: Cluster): Either[String, Cluster] = 
  Right(Cluster(cluster.pods))
```

We can compose them in same maner as we have done with **Option** example above:

```scala
val ns = "my-cluster"
for {
   _ <- validateNamespace(ns)
   _ <- clusterExists(ns).left.map(c => 
           s"Cluster with ${c.pods} pods already exists")
   newCluster <- createCluster(ns, Cluster(4))
} yield newCluster
```

From business logic perspective we want to create some hypothetical cluster if namespace is valid and cluster for the given namespace does not exist. 
We modeled errors as `Either.Left` and normal result as `Either.Right`. It is popular approach not only in Scala to have some sort of result wrapper.

Result value is based on the function values we hardcoded in the given functions:

```scala
scala> Either[String,Cluster] = Right(Cluster(4))
```

Benefits of using Monads is that we do not need to use `if/else` control flow, since we have Monads Laws working when we compose Monad instances.

In case some of the given functions returns Either.Left:

```scala
def validNamespace(ns: String): Either[String, Unit] = 
   if (ns == "my-cluster") 
   Left(
     “Cluster namespace is not valid name, choose another name”
   ) else Right(())
```   

Then it turns the whole result of the composition into error state, i.e. Either.Left:

```scala
scala> Either[String,Cluster] = Left(
              Cluster namespace is not valid name, choose another name
            )
```            

## For comprehension

Scala offers special syntax for the sequence of nested `flatMap` calls and one `map` at the end, which is called "for-comprehension".

**for {…} yield** is a syntactic sugar for a sequence of calls:

```scala
flatMap1(… + flatMapN(.. + map(…)))
```

**Desugared version**:

```scala
validNamespace("my-cluster")
  .flatMap(_ =>
     clusterExists(ns)
       .left
       .map(c => s"Cluster with ${c.pods} pods already exists")
       .flatMap(_ =>
            createCluster(ns, Cluster(4))
               .map(newCluster => newCluster)
        )
  )
```

**Sugared version of the same code snippet**:

```scala
for {
  _ <- validNamespace("my-cluster")
  _ <- clusterExists(ns).left.map(c => 
          s"Cluster with ${c.pods} pods already exists")
  newCluster <- createCluster(ns, Cluster(4))
} yield newCluster

```

 We can easilly compose only the Monad type, in order to compose different Monad types, we can use additional technique called Monad Transformers.