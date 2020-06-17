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
a user does not provide any custom value. In the example below, I apply scalaboot template for a new
project called `app-1`. I also set my project description and newer Scala version to be specified in the 
build.sbt:

```bash
[info] Loading settings for project global-plugins from idea.sbt,gpg.sbt ...
[info] Loading global plugins from /Users/alexey/.sbt/1.0/plugins
[info] Set current project to git (in build file:/Users/alexey/dev/git/)
name [Project Name]: app-1
desc [Describe your project a bit]: order registration
scalaVersion [2.13.1]: 2.13.2

Template applied in /Users/alexey/dev/git/./app-1
```