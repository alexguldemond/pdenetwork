package org.alexguldemond.pdenetwork.model

import breeze.linalg.{DenseMatrix, DenseVector, Transpose}
import breeze.numerics.constants.Pi
import breeze.numerics.sin
import org.alexguldemond.pdenetwork.network.{SimpleNetwork, WeightVector}

case class SineBcLaplacianModel(simpleNetwork: SimpleNetwork) extends SimpleLaplacianModel(simpleNetwork) {
  import SineBcLaplacianModel._

  override def bcSatisfier(input: DenseVector[Double]): Double = {
    val x1 = input(0)
    val x2 = input(1)
    x2*sin(Pi * x1)
  }

  override def diffOpBcSatisfier(input: DenseMatrix[Double]): Transpose[DenseVector[Double]] = {
    val x1 = input(0,::)
    val x2 = input(1,::)
    x2 *:* sin(x1*Pi)*minusPiSquared
  }

  override def data(input: DenseMatrix[Double]): Transpose[DenseVector[Double]] = {
    DenseVector.zeros[Double](input.cols).t
  }

  override def copyArchitecture(weightVector: DenseVector[Double]): Model = {
    val newNetwork = SimpleNetwork(
      WeightVector.vecToWeightGrad(weightVector, 2, simpleNetwork.innerWeights.rows),
      simpleNetwork.activation)

    SineBcLaplacianModel(newNetwork)
  }
}

object SineBcLaplacianModel {
  lazy val minusPiSquared = -Pi*Pi
}