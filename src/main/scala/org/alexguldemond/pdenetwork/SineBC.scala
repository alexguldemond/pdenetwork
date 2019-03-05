package org.alexguldemond.pdenetwork

import breeze.numerics.constants.Pi
import breeze.numerics.{exp, sin}
import org.alexguldemond.pdenetwork.plot.BreezePlotting._
import org.alexguldemond.pdenetwork.plot.Jzy3dPlotting._
import org.alexguldemond.pdenetwork.plot.Plot._
import org.alexguldemond.pdenetwork.utils.Timer.time
import org.alexguldemond.pdenetwork.mesh.Uniform2DMesh
import org.alexguldemond.pdenetwork.model.{Model, SineBcLaplacianModel}
import org.alexguldemond.pdenetwork.network.SimpleNetwork
import org.alexguldemond.pdenetwork.updater.AdamUpdater
import org.jzy3d.plot3d.builder.Mapper

import scala.collection.mutable.ListBuffer

object SineBC extends App {

  val mesh = Uniform2DMesh(.02)
  val model: Model = SineBcLaplacianModel(SimpleNetwork.randomNetwork(2, 10, mesh.numberOfPoints))

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
      report ++= model.fit(iter, reportFrequency, AdamUpdater(0.01))
      println(s"Epoch $i complete")
    }
  }

  model.plot("Approximate Solution to u_xx + u_yy = 0")
  exact.plot("Exact Solution to u_xx + u_yy = 0")
  report.toList.plot("Error")


}
