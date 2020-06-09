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
tool in the Scala eco-system and the most used one among Scala developers. I am using SBT already for many years and found the following plugins which I use in most the projects:

## Giter8

It allows to create an SBT project template to be used to ramp up new SBT project. Project template usually includes typical configuration
an SBT user copy paster from project to project. User can put any file into template.

One part of the Giter8 is embeed into SBT. The second part is the Giter8 plugin itself. User can create new SBT project from template hosted at GitHub and that the most convenient part. You just an internet access and run SBT `new` command. For example:

```scala
sbt new novakov-alexey/scalaboot.g8     
```

scalaboot.g8 is a name of the GitHub repository at my personal account `novakov-alexey`. Actually, SBT looks for repository at 
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