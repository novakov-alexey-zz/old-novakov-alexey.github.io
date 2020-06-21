+++
title="SBT Plugins"
date=2020-06-29
draft = true

[extra]
category="blog"

[taxonomies]
tags = ["sbt", "plugins"]
categories = ["scala"]
+++

SBT is a Scala Build Tool. It is written in Scala and can compile, build artefacts for Scala and Java projects. SBT is also the first build
tool in the Scala eco-system and the most used one among Scala developers. I am using SBT already for many years and found the following useful plugins which I use in most my projects:

## Giter8

It allows to create an SBT project template to be used to ramp up new SBT project. Project template usually includes typical configuration that 
an SBT user copy and paste from project to project. User can put any file into template.

One part of the Giter8 is embeed into SBT. The second part is the Giter8 plugin itself. User can create new SBT project from template hosted at GitHub and that the most useful part. You just need an internet access, then run SBT `new` command. For example:

```scala
sbt new novakov-alexey/scalaboot.g8     
```

scalaboot.g8 is a name of the GitHub repository at my personal account `novakov-alexey`. SBT converts template name into  
https://github.com/novakov-alexey/scalaboot.g8 path, which you can visit in browser as well. There is also an option to use Giter8 template from the local file system.

Once we run above command, SBT Giter8 plugin will create new project file structure such as:

```bash
├── build.sbt
├── project
│   ├── Dependencies.scala
│   ├── build.properties
│   └── plugins.sbt
├── src
│   ├── main
│   └── test
└── version.sbt
```

### Template File Structure

Project template is also a SBT project :-). In case of scalaboot.g8 example, it looks like this:

{{ resize_image(path="sbt-plugins/scalaboot.png", width=600, height=600, op="fit") }}

Content of the g8 folder is a template content which will be used when users apply this template
for their projects. Giter8 template supports propertis and special syntax for them. Properties can
be put in any text file. Let's look at the example in build.sbt file:

```scala
ThisBuild / organization := "com.example"
ThisBuild / scalaVersion := "$scalaVersion$"
ThisBuild / description  := "$desc$"

lazy val root = (project in file(".")).
  settings(
    name := "$name;format="lower,hyphen"$",
    libraryDependencies ++= Seq(
      akkaHttp,
      akkaStreams,
      scalaLogging,
      logback,
      …

```

A text between `$ ... $` is evaluated by Giter8 and replaced by pre-defined or user-given parameters.
In my scalaboot template, I have such pre-defined values in properties file:

*src/main/g8/default.properties*

```properties
name=Project Name
desc=Describe your project a bit
scalaVersion=2.13.1
```

SBT `new` command is going through the list of defined properties and sets the default values in case
users do not provide their own values. In the example below, I apply `scalaboot` template for
project name `app-1`. I also set my project description and newer Scala version. They will apear in the build.sbt:

```bash
[info] Loading settings for project global-plugins from idea.sbt,gpg.sbt ...
[info] Loading global plugins from /Users/alexey/.sbt/1.0/plugins
[info] Set current project to git (in build file:/Users/alexey/dev/git/)
name [Project Name]: app-1
desc [Describe your project a bit]: order registration
scalaVersion [2.13.1]: 2.13.2

Template applied in /Users/alexey/dev/git/./app-1
```

## SBT Revolver

It is simple plugin to be added in a Scala project. However, Gitter8 is not necesserily to be added in your project, only when you
develop new template and want to keep in the GitHub repo. Revolver plugin can be added to project as any other user pluging via project/plugins.sbt file.
Adding its definition to that file:

```scala
addSbtPlugin("io.spray" % "sbt-revolver" % “x.y.z")
```

Change x.y.z to latest version from its GitHub repository. 

Main feature of sbt-reolver is triggered execution upon project file modification. It helps to restart your application automatically and may remind
you dynamic-language expereince where developers test their modules by refreshing the browser page or by calling their scripts again.

{{ resize_image(path="sbt-plugins/fe-dev.png", width=600, height=600, op="fit") }}

In summary, *it enables a super-fast development turnaround for your Scala applications*.

SBT Reolver starts your application in fored JVM, that helps to easily pass JVM options and restart it again upon triggered execution.

{{ resize_image(path="sbt-plugins/revolver-jvm.png", width=600, height=600, op="fit") }}

*MainExchange* is my Scala application started by sbt-revolver as separate JVM process.

Revolver has its own configuration to control JVM options, environement variables, etc.

JVM options example:
```scala
javaOptions in reStart += "-Xmx2g"
```

Set your main class if you have more than one:
```scala
mainClass in reStart := Some("com.example.Main")
```

Enable debug of the forked JVM process:
```scala
Revolver.enableDebugging(port = 5050, suspend = true)
```

Export environment variables for your Scala application:
```scala
envVars in reStart := Map(“K8S_NAMESPACE" -> “test")
```

Let us at below example, when we start our application via sbt-revolver:

{{ resize_image(path="sbt-plugins/revolver-start.png", width=600, height=600, op="fit") }}

MainExchange is Akka-HTTP based application running HTTP server. Now let us change some line of code in the code base.
Once we done that, sbt-revolver immediatelly triggers compilation, stop running process and starts new one:

{{ resize_image(path="sbt-plugins/revolver-triggered.png", width=600, height=600, op="fit") }}

There are 3 things happened:

- Build triggered (compilation)
- Stop running application
- Start new application

`restart` revolver SBT task is levegaring SBT triggered execution which is enabled by tilda *~* in front the task name,
 when running it in SBT shell.

 There are other useful commands to be combined with ~ to trigger some task upon files modification:

 ```bash
// runs failed tests, if any
~ testQuick

// runs specific test
~ testOnly org.alexeyn.SomeTest

// runs all tests
~ test

// cleans compiled sources and runs all tests
~ clean; test
 ```

 ## SBT Tpolecat

 Enables Scala compiler options as per recommendatons of Rob Norris [blog-post](https://tpolecat.github.io/2017/04/25/scalac-flags.html). Plugin enables as many Scala compiler options as possible to enforce type safety and
 discorage bad practises in a code base by turning warnings into compiler errors.

Add plungin to your project:

 ```scala
 addSbtPlugin(“io.github.davidgregory084" % "sbt-tpolecat" % “0.1.10")
 ```

 Actually, the same compiler options can be enabled manually within the SBT definition. However, it is more convenient to
 enable this plugin once and forget about adding anything mnually. One can also disbale particular options enabled by this
 plugin, in case that option does not make sense for particular project.

 Some of the options which are enabled by this plugin:

 ```scala
 scalacOptions ++= Seq(
  "-deprecation",               
  "-encoding", "utf-8",         
  "-explaintypes",                  
  "-language:higherKinds",          
  "-language:implicitConversions",  
  "-unchecked",                       
  "-Xfatal-warnings",            
  "-Xlint:infer-any",                 
  "-Ywarn-dead-code",              
  "-Ywarn-extra-implicit",        
  "-Ywarn-inaccessible",          
  "-Ywarn-infer-any",            
  "-Ywarn-numeric-widen",       
  "-Ywarn-unused:implicits",   
  "-Ywarn-unused:imports",     
  "-Ywarn-unused:locals",     
  "-Ywarn-unused:params",     
  "-Ywarn-unused:patvars",    
  "-Ywarn-unused:privates",   
  "-Ywarn-value-discard"     
…    
)
 ```

Last I checked this plugin it enables 54 scalac options. I recommend to use this plugin by default in every project, it will
make your code base much more robust.

## SBT Native Packager

To enbale in your project:

```scala
addSbtPlugin("com.typesafe.sbt" %% "sbt-native-packager" % “x.y.z")
```

Native Packager allows to package your application in different formats such as:
- universal zip, tar.gz, xz archives
- deb and rpm packages
- dmg 
- msi 
- Docker
- GraalVM native images

Native packager is not auto-plugin, i.e. it is not enabled by default. In order to use it for some of
your module, you need to enable it in SBT definition:

```scala
lazy val root = (project in file(".")).
  settings(
    name := "exchange",
    ….
    dockerBaseImage := “openjdk:8-jre-alpine”,
    dockerExposedPorts ++= Seq(8080),
    dockerRepository := Some(“alexeyn")
  ).enablePlugins(AshScriptPlugin) 
                          // or other options - DockerPlugin, JavaAppPackaging
```
This plugins comes with different types packaging which you can choose when enabling it for some SBT module.
In the example above, we enable Java packaging format with Ash shell comptabile executable script, so that we can run a jar file
in Alpine Linux. Basically, JavaAppPacking is a base format which is creating a couple of scripts to start JVM with long list 
of JAR files in the CLASSPATH variable. It also puts all required depednecies into the `lib` folder, 
which is referenced from that automatically generated shell script. 

### Java Packaging Format

Below an example of such SBT task. It builds a ZIP archive, which is kind of universal assuming that built application is running in JVM. 

```scala
sbt universal:packageBin
```

it will create a zip archive with a file structure shown below:

```bash
~/dev/git/exchange/target/universal/exchange-0.1.1-SNAPSHOT.zip

tree -L 2
├── bin
│   ├── exchange
│   └── exchange.bat
└── lib
    ├── ch.qos.logback.logback-classic-1.2.3.jar
    ├── ch.qos.logback.logback-core-1.2.3.jar
    ├── com.chuusai.shapeless_2.13-2.3.3.jar
    ├── com.example.exchange-0.1.1-SNAPSHOT.jar
    ├── com.google.protobuf.protobuf-java-3.10.0.jar
    ├── com.typesafe.akka.akka-actor_2.13-2.6.1.jar
    ├── com.typesafe.akka.akka-http-core_2.13-10.1.11.jar
    ├── com.typesafe.akka.akka-http_2.13-10.1.11.jar
    ├── com.typesafe.akka.akka-parsing_2.13-10.1.11.jar
    ├── com.typesafe.akka.akka-protobuf-v3_2.13-2.6.1.jar
    ├── com.typesafe.akka.akka-stream_2.13-2.6.1.jar
    ├── com.typesafe.config-1.4.0.jar
    ├── com.typesafe.scala-logging.scala-logging_2.13-3.9.2.jar
    ├── com.typesafe.ssl-config-core_2.13-0.4.1.jar
    ├── de.heikoseeberger.akka-http-circe_2.13-1.30.0.jar
    ├── io.circe.circe-core_2.13-0.12.3.jar
    ├── io.circe.circe-generic_2.13-0.12.3.jar
    ├── io.circe.circe-jawn_2.13-0.12.3.jar
    ├── io.circe.circe-numbers_2.13-0.12.3.jar
    ├── io.circe.circe-parser_2.13-0.12.3.jar
    ├── org.reactivestreams.reactive-streams-1.0.3.jar
    ├── org.scala-lang.modules.scala-java8-compat_2.13-0.9.0.jar
    ├── org.scala-lang.modules.scala-parser-combinators_2.13-1.1.2.jar
    ├── org.scala-lang.scala-library-2.13.1.jar
    ├── org.scala-lang.scala-reflect-2.13.1.jar
    ├── org.slf4j.slf4j-api-1.7.26.jar
    ├── org.typelevel.cats-core_2.13-2.0.0.jar
    ├── org.typelevel.cats-kernel_2.13-2.0.0.jar
    ├── org.typelevel.cats-macros_2.13-2.0.0.jar
    └── org.typelevel.jawn-parser_2.13-0.14.2.jar

2 directories, 32 files
```

*bin/exchange* is a shell script to run your Scala application Main class.

### Docker Image format

SBT task to create a Dockerfile and the same file structure as for Java packaging format:

```scala
sbt docker:stage
```

```bash
cd /Users/alexey/dev/git/exchange/target/docker
tree -L 5
.
└── stage
    ├── Dockerfile
    └── opt
        └── docker
            ├── bin
            │   ├── exchange
            │   └── exchange.bat
            └── lib
                ├── ch.qos.logback.logback-classic-1.2.3.jar
                ├── ch.qos.logback.logback-core-1.2.3.jar
                ├── com.chuusai.shapeless_2.13-2.3.3.jar
                ├── com.example.exchange-0.1.1-SNAPSHOT.jar
```                

In order to build an image and publish it to a container registry:

```scala
sbt docker:publish
```

You can also customize Dockerfile, which is by default generated automatically. 
Default docker file content can be be seen via:

```scala
sbt> show dockerCommands

[info] * Cmd(FROM,WrappedArray(openjdk:8, as, stage0))
[info] * Cmd(LABEL,WrappedArray(snp-multi-stage="intermediate"))
[info] * Cmd(LABEL,WrappedArray(snp-multi-stage-id="b8437d6f-af0a-459c-ae51-cd3b9c5b7404"))
[info] * Cmd(WORKDIR,WrappedArray(/opt/docker))
[info] * Cmd(COPY,WrappedArray(opt /opt))
[info] * Cmd(USER,WrappedArray(root))
[info] * ExecCmd(RUN,List(chmod, -R, u=rX,g=rX, /opt/docker))
[info] * ExecCmd(RUN,List(chmod, u+x,g+x, /opt/docker/bin/exchange))
```

In order to customize Dockerfile content:

```scala
dockerCommands := Seq(
  Cmd("FROM", "openjdk:8"),
  Cmd("LABEL", s"""MAINTAINER="${maintainer.value}""""),
  ExecCmd("CMD", "echo", "Hello, World from Docker")
)
```

## SBT Release