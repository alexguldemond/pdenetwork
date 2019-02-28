name := "pdenetwork"

version := "0.1"

scalaVersion := "2.12.8"

// https://mvnrepository.com/artifact/org.scalanlp/breeze
libraryDependencies += "org.scalanlp" %% "breeze" % "0.13.2"

// https://mvnrepository.com/artifact/org.scalanlp/breeze-natives
libraryDependencies += "org.scalanlp" %% "breeze-natives" % "0.13.2"

resolvers += "Jzy3d releases" at "http://maven.jzy3d.org/releases/"

//https://mvnrepository.com/artifact/org.jzy3d/jzy3d-api
libraryDependencies += "org.jzy3d" % "jzy3d-api" % "1.0.2"

// https://mvnrepository.com/artifact/org.scalanlp/breeze-viz
libraryDependencies += "org.scalanlp" %% "breeze-viz" % "1.0-RC2"

// https://mvnrepository.com/artifact/com.typesafe.scala-logging/scala-logging
libraryDependencies += "com.typesafe.scala-logging" %% "scala-logging" % "3.9.2"

// https://mvnrepository.com/artifact/ch.qos.logback/logback-classic
libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.2.3" % Test

libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.0-SNAP10" % Test

// https://mvnrepository.com/artifact/org.scalacheck/scalacheck
libraryDependencies += "org.scalacheck" %% "scalacheck" % "1.14.0" % Test
