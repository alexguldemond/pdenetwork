package org.alexguldemond.pdenetwork.model

import breeze.linalg.{DenseMatrix, DenseVector, Transpose}
import org.alexguldemond.pdenetwork.network.SimpleNetwork

abstract class ZeroBcLaplacianModel(simpleNetwork: SimpleNetwork) extends SimpleLaplacianModel(simpleNetwork) {
  override def bcSatisfier(input: DenseVector[Double]): Double = 0d

  override def diffOpBcSatisfier(input: DenseMatrix[Double]): Transpose[DenseVector[Double]] =
    DenseVector.zeros[Double](input.cols).t

}
