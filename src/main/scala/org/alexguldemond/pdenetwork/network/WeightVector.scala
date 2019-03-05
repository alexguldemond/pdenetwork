package org.alexguldemond.pdenetwork.network

import breeze.linalg.{DenseMatrix, DenseVector}

case class WeightVector(innerWeights: DenseMatrix[Double],
                        innerBias: DenseVector[Double],
                        outerWeight: DenseVector[Double]) {

  def *(scalar: Double) = WeightVector(scalar * innerWeights,
    scalar * innerBias, scalar * outerWeight)

  def :*=(scalar: Double): WeightVector = {
    innerWeights :*= scalar
    innerBias :*= scalar
    outerWeight :*= scalar
    this
  }

  def /(scalar: Double) = WeightVector(innerWeights/scalar,
    innerBias/scalar, outerWeight/scalar)

  def :/=(scalar: Double): WeightVector = {
    innerWeights :/= scalar
    innerBias :/= scalar
    outerWeight :/= scalar
    this
  }

  def +(other: WeightVector): WeightVector =
    WeightVector(innerWeights + other.innerWeights,
      innerBias + other.innerBias,
      outerWeight + other.outerWeight)

  def :+=(other: WeightVector): WeightVector = {
    innerWeights :+= other.innerWeights
    innerBias :+= other.innerBias
    outerWeight :+= other.outerWeight
    this
  }

  def toDenseVector = WeightVector.weightGradToVec(this)

}

object WeightVector {
  def weightGradToVec(weightGradient: WeightVector): DenseVector[Double] = {

    DenseVector.vertcat(weightGradient.innerWeights.flatten(),
      weightGradient.innerBias,
      weightGradient.outerWeight)
  }

  def vecToWeightGrad(vec: DenseVector[Double], inputSize: Int, hiddenSize: Int): WeightVector = {
    WeightVector(vec(0 to inputSize * hiddenSize - 1).asDenseMatrix.reshape(hiddenSize, inputSize),
      vec(inputSize * hiddenSize to (inputSize + 1)* hiddenSize -1),
      vec((inputSize + 1)* hiddenSize to (inputSize + 2)* hiddenSize -1))
  }
}

