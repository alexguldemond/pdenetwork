package org.alexguldemond.pdenetwork

import breeze.linalg.{DenseMatrix, DenseVector, Transpose}

trait Network extends NetworkDerivative {
  def inputDerivative(multiIndex: MultiIndex) : NetworkDerivative

  def updateWeights(weightGradient: WeightGradient): Network
}

trait NetworkDerivative {
  def apply(input: DenseVector[Double]): Double

  def applyBatch(input: DenseMatrix[Double]): Transpose[DenseVector[Double]]

  def weightGradient(input: DenseVector[Double]) : WeightGradient

  def weightGradientBatch(input: DenseMatrix[Double]) : WeightGradientBatch
}

