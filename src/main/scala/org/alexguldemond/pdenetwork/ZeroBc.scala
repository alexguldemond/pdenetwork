package org.alexguldemond.pdenetwork

import breeze.linalg.{DenseMatrix, DenseVector, Transpose}
import breeze.numerics.constants._
import breeze.numerics.sin
import org.alexguldemond.pdenetwork.mesh.Uniform2DMesh
import org.alexguldemond.pdenetwork.model.{Model, ZeroBcLaplacianModel}
import org.alexguldemond.pdenetwork.network.SimpleNetwork
import org.alexguldemond.pdenetwork.plot.BreezePlotting._
import org.alexguldemond.pdenetwork.plot.Jzy3dPlotting._
import org.alexguldemond.pdenetwork.plot.Plot._
import org.alexguldemond.pdenetwork.updater.BFGSUpdater
import org.alexguldemond.pdenetwork.utils.Timer.time

import scala.collection.mutable.ListBuffer

object ZeroBc extends App{
  val mesh = Uniform2DMesh(.01)

  val model: Model = new ZeroBcLaplacianModel(SimpleNetwork.randomNetwork(2, 15)) {
    override def data(input: DenseMatrix[Double]): Transpose[DenseVector[Double]] = {
      val x1 = input(0, ::)
      val x2 = input(1, ::)
      sin(x1*Pi*4d)*:*sin(x2*Pi*4d)
      DenseVector.fill(input.cols, -1d).t
    }
  }

  val epochs = 20
  val batchSize = 10
  val reportFrequency = 20
  var report: ListBuffer[Double] = ListBuffer[Double]()
  val updater = BFGSUpdater(.01, 1000, 1)

  time {
    for (i <- 1 to epochs) {
      val iter = mesh.iterator(batchSize)
      report ++= model.fit(iter, reportFrequency, updater)
      println(s"Epoch $i complete")
    }
  }

  model.plot("Approximate Solution to u_xx + u_yy = 1")
  report.toList.plot("Error")

}
