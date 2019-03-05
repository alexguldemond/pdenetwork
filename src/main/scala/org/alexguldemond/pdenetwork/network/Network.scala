package org.alexguldemond.pdenetwork.network

import breeze.linalg.{DenseMatrix, DenseVector, Transpose}

trait Network extends NetworkDerivative {
  def inputDerivative(multiIndex: MultiIndex) : NetworkDerivative

  def updateWeights(weightGradient: WeightVector): Network

  def weightVector: WeightVector
}

trait NetworkDerivative {
  def apply(input: DenseVector[Double]): Double

  def applyBatch(input: DenseMatrix[Double]): Transpose[DenseVector[Double]]

  def weightGradient(input: DenseVector[Double]) : WeightVector

  def weightGradientBatch(input: DenseMatrix[Double]) : WeightGradientBatch
}

