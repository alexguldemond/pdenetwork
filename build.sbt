name := "pdenetwork"

version := "0.1"

scalaVersion := "2.12.8"

// https://mvnrepository.com/artifact/org.scalanlp/breeze
libraryDependencies += "org.scalanlp" %% "breeze" % "0.13.2"

// https://mvnrepository.com/artifact/org.scalanlp/breeze-natives
libraryDependencies += "org.scalanlp" %% "breeze-natives" % "0.13.2"

resolvers += "Jzy3d releases" at "http://maven.jzy3d.org/releases/"

// https://mvnrepository.com/artifact/org.jzy3d/jzy3d-api
libraryDependencies += "org.jzy3d" % "jzy3d-api" % "1.0.2"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.0-SNAP10" % Test

// https://mvnrepository.com/artifact/org.scalacheck/scalacheck
libraryDependencies += "org.scalacheck" %% "scalacheck" % "1.14.0" % Test
