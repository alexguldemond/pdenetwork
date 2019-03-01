package org.alexguldemond.pdenetwork

import breeze.linalg.{DenseMatrix, DenseVector, Transpose}

case class GaussianLaplacianModel(simpleNetwork: SimpleNetwork) extends SimpleLaplacianModel(simpleNetwork) {
  override def bcSatisfier(input: DenseVector[Double]): Double = 0d

  override def diffOpBcSatisfier(input: DenseMatrix[Double]): Transpose[DenseVector[Double]] =
    DenseVector.zeros[Double](input.cols).t

  override def data(input: DenseMatrix[Double]): Transpose[DenseVector[Double]] = DenseVector.fill(input.cols, 1d).t
}
