package org.alexguldemond.pdenetwork

import breeze.numerics.constants.Pi
import breeze.numerics.{exp, sin}
import org.alexguldemond.pdenetwork.Timer.time
import org.alexguldemond.pdenetwork.BreezePlotting._
import org.alexguldemond.pdenetwork.Jzy3dPlotting._
import org.alexguldemond.pdenetwork.Plot._
import org.jzy3d.plot3d.builder.Mapper

import scala.collection.mutable.ListBuffer

object Gaussian extends App{
  val mesh = Uniform2DMesh(.02)
  val model: Model = GaussianLaplacianModel(SimpleNetwork.randomNetwork(2, 10))

  val epochs = 15
  val batchSize = 10
  val reportFrequency = 5
  var report: ListBuffer[Double] = ListBuffer[Double]()

  val exact = new Mapper() {
    def f(x: Double, y: Double): Double = {
      sin(Pi*x)*(exp(Pi*y) - exp(-Pi*y))/(exp(Pi)-exp(-Pi))
    }
  }

  time {
    for (i <- 1 to epochs) {
      val iter = mesh.iterator(batchSize)
      report ++= model.fit(iter, reportFrequency, SGDUpdater(learningRate = .5))
      println(s"Epoch $i complete")
    }
  }

  model.plot("Approximate Solution to u_xx + u_yy = 1")
  report.toList.plot("Error")

}
