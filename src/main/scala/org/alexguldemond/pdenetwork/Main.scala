package org.alexguldemond.pdenetwork

import org.alexguldemond.pdenetwork.Jzy3dPlotting._
import BreezePlotting._
import org.alexguldemond.pdenetwork.Plot._

import scala.collection.mutable.ListBuffer

object Main extends App {

  val mesh = Uniform2DMesh(.02)
  val model: SimpleLaplacianModel = SimpleLaplacianModel.randomModel(10)

  val epochs = 15
  val batchSize = 10
  val reportFrequency = 5
  var report: ListBuffer[Double] = ListBuffer[Double]()

  time {
    for (i <- 1 to epochs) {
      val iter = mesh.iterator(batchSize)
      report ++= model.fit(iter, reportFrequency, SGDUpdater(learningRate = .01))
      println(s"Epoch $i complete")
    }
  }

  model.plot("Approximate Solution to u_xx + uyy = 0")
  report.toList.plot("Error")

  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) + "ns")
    result
  }

}
