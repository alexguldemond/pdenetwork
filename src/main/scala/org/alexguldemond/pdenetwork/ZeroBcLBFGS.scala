package org.alexguldemond.pdenetwork

import breeze.linalg.{DenseVector, Transpose}
import breeze.numerics.constants.Pi
import breeze.numerics.sin
import org.alexguldemond.pdenetwork.activation.ArcTan
import org.alexguldemond.pdenetwork.mesh.Uniform2DMesh
import org.alexguldemond.pdenetwork.model.{Model, ZeroBcLaplacianModel}
import org.alexguldemond.pdenetwork.network.SimpleNetwork
import org.alexguldemond.pdenetwork.plot.Jzy3dPlotting._
import org.alexguldemond.pdenetwork.plot.Plot._
import org.alexguldemond.pdenetwork.updater.LBFGSUpdater
import org.alexguldemond.pdenetwork.utils.Timer.time

object ZeroBcLBFGS extends App{

  val mesh = Uniform2DMesh(.02)

  val model: Model =
    ZeroBcLaplacianModel(SimpleNetwork.randomNetwork(2, 10, 1, ArcTan)) { x =>
      val x1: Transpose[DenseVector[Double]] = x(0, ::)
      val x2 = x(1,::)
      sin(x1 * Pi * 4d)
    }

  val updater = LBFGSUpdater(100, 40)
  val optimumModel = time {
    model.update(mesh.allPoints, updater)
  }

  optimumModel.plot("Approx Solution")

}
