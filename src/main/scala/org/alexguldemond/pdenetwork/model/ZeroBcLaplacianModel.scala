package org.alexguldemond.pdenetwork.model

import breeze.linalg.{DenseMatrix, DenseVector, Transpose}
import org.alexguldemond.pdenetwork.network.{SimpleNetwork, WeightVector}

abstract class ZeroBcLaplacianModel(simpleNetwork: SimpleNetwork) extends SimpleLaplacianModel(simpleNetwork) {
  override def bcSatisfier(input: DenseVector[Double]): Double = 0d

  override def diffOpBcSatisfier(input: DenseMatrix[Double]): Transpose[DenseVector[Double]] =
    DenseVector.zeros[Double](input.cols).t

  override def copyArchitecture(weightVector: DenseVector[Double]): Model = {
    val newNetwork = SimpleNetwork(
      WeightVector.vecToWeightGrad(weightVector, 2, simpleNetwork.innerWeights.rows),
      simpleNetwork.activation)

    ZeroBcLaplacianModel(newNetwork)(this.data _)
  }

}

object ZeroBcLaplacianModel {

  def apply (simpleNetwork: SimpleNetwork)
            (dataFunc: DenseMatrix[Double] => Transpose[DenseVector[Double]]): ZeroBcLaplacianModel = {
    new ZeroBcLaplacianModel(simpleNetwork) {
      override def data(input: DenseMatrix[Double]): Transpose[DenseVector[Double]] = dataFunc(input)
    }
  }

}
