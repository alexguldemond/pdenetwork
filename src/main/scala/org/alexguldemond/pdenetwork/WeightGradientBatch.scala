package org.alexguldemond.pdenetwork

import breeze.linalg._

case class WeightGradientBatch(innerWeightGradients: Seq[DenseMatrix[Double]],
                               innerBiasGradients: DenseMatrix[Double],
                               outerWeightGradients: DenseMatrix[Double]) {

  def length = innerWeightGradients.size

  def apply(i: Int) = WeightGradient(innerWeightGradients(i), innerBiasGradients(::,i), outerWeightGradients(::,i))

  def dotSum(vec: DenseVector[Double]): WeightGradient = {

    val outerWeightSum: DenseVector[Double] = outerWeightGradients(*,::) dot vec
    val innerBiasSum: DenseVector[Double] = innerBiasGradients(*,::) dot vec

    val innerWeightSum: DenseMatrix[Double] =
      DenseMatrix.zeros[Double](innerWeightGradients(0).rows, innerWeightGradients(0).cols)

    for ( (scalar, mat) <- vec.toScalaVector.zip(innerWeightGradients)) {
      innerWeightSum :+= (scalar * mat)
    }

    WeightGradient(innerWeightSum, innerBiasSum, outerWeightSum)
  }

  def dotSum(vec: Transpose[DenseVector[Double]]): WeightGradient = dotSum(vec.t)

  def elemProd(vec: DenseVector[Double]): WeightGradientBatch = {
    val outerWeights: DenseMatrix[Double] = outerWeightGradients(*,::) *:* vec
    val innerBiases: DenseMatrix[Double] = innerBiasGradients(*,::) *:* vec
    val innerWeights: Seq[DenseMatrix[Double]] =
      for ( (w, scalar) <- innerWeightGradients zip vec.toScalaVector()) yield {
        w*scalar
      }
    WeightGradientBatch(innerWeights, innerBiases, outerWeights)
  }

  def elemProd(vec: Transpose[DenseVector[Double]]): WeightGradientBatch = elemProd(vec.t)

  def prod(scalar: Double): WeightGradientBatch =
    WeightGradientBatch(innerWeightGradients.map{_ * scalar}, innerBiasGradients * scalar, outerWeightGradients * scalar)

  def inPlaceProd(scalar: Double): WeightGradientBatch = {
    innerWeightGradients.foreach{_ :*= scalar}
    innerBiasGradients :*= scalar
    outerWeightGradients :*= scalar
    this
  }

  def sum(other: WeightGradientBatch): WeightGradientBatch =
    WeightGradientBatch( for ((w, ow) <- innerWeightGradients zip other.innerWeightGradients) yield {
      w + ow
    },
      innerBiasGradients + other.innerBiasGradients,
      outerWeightGradients + other.outerWeightGradients)

  def inPlaceSum(other: WeightGradientBatch) = {

    for ((w, ow) <- innerWeightGradients zip other.innerWeightGradients) {
      w :+= ow
    }

    innerBiasGradients :+= other.innerBiasGradients
    outerWeightGradients :+= other.outerWeightGradients
    this
  }
}
