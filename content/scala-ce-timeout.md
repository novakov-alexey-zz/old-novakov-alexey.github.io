+++
title="First Post"
date=2020-02-26
draft = true

[extra]
category="blog"
+++

import java.util.concurrent.Executors

import cats.effect.ExitCase._
import cats.effect.{Blocker, Concurrent, ContextShift, IO, Timer}
import cats.implicits._

import scala.concurrent.ExecutionContext
import scala.concurrent.duration._
import scala.sys.process._

implicit val timer: Timer[IO] = IO.timer(ExecutionContext.global)
implicit val cs: ContextShift[IO] = IO.contextShift(ExecutionContext.global)
val cachedThreadPool = Executors.newCachedThreadPool()
val blocker = Blocker.liftExecutionContext(ExecutionContext.fromExecutor(cachedThreadPool))

val blockingTask = blocker.blockOn(IO("tail -f build.sbt".run()))
val task = blockingTask.bracketCase { p =>
    IO(p.exitValue())
  } { (p, exit) =>
    exit match {
      case Completed => IO.unit
      case Error(_) | Canceled => IO(p.destroy())
    }
  }

def timeoutTo[F[_], A](
    fa: F[A],
    after: FiniteDuration,
    fallback: F[A]
  )(implicit timer: Timer[F], cs: ContextShift[F], concurrent: Concurrent[F]): F[A] = {

    concurrent.race(fa, timer.sleep(after)).flatMap {
      case Left(a) =>
        println("Done")
        concurrent.pure(a)
      case Right(_) =>
        println("Timeout")
        fallback
    }
  }

val finalTask = timeoutTo(task, 1.second, IO("fallback"))
finalTask.unsafeRunSync()
