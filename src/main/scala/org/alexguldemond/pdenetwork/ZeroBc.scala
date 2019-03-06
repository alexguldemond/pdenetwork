package org.alexguldemond.pdenetwork

import breeze.linalg.{DenseVector, Transpose}
import breeze.numerics.sin
import breeze.numerics.constants.Pi
import org.alexguldemond.pdenetwork.activation.SoftPlus
import org.alexguldemond.pdenetwork.mesh.Uniform2DMesh
import org.alexguldemond.pdenetwork.model.{Model, ZeroBcLaplacianModel}
import org.alexguldemond.pdenetwork.network.SimpleNetwork
import org.alexguldemond.pdenetwork.plot.BreezePlotting._
import org.alexguldemond.pdenetwork.plot.Jzy3dPlotting._
import org.alexguldemond.pdenetwork.plot.Plot._
import org.alexguldemond.pdenetwork.updater.AdamUpdater
import org.alexguldemond.pdenetwork.utils.Timer.time

import scala.collection.mutable.ListBuffer

object ZeroBc extends App{
  val mesh = Uniform2DMesh(.02)

  val model: Model =
    ZeroBcLaplacianModel(SimpleNetwork.randomNetwork(2, 10, 1, SoftPlus)) { x =>
      val x1: Transpose[DenseVector[Double]] = x(0, ::)
      val x2 = x(1,::)
      sin(x1 * Pi * 4d)
  }

  val epochs = 20
  val batchSize = 10
  val reportFrequency = 2
  var report: ListBuffer[Double] = ListBuffer[Double]()
  val updater = AdamUpdater(0.01)

  time {
    for (i <- 1 to epochs) {
      report ++= model.fit(mesh, batchSize,reportFrequency, updater)
      println(s"Epoch $i complete")
    }
  }

  model.plot("Approximate Solution to u_xx + u_yy = 1")
  report.toList.plot("Error")

}
