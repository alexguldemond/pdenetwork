package org.alexguldemond.pdenetwork

import breeze.linalg.{DenseMatrix, DenseVector}

case class WeightGradient(innerWeightGradient: DenseMatrix[Double],
                          innerBiasGradient: DenseVector[Double],
                          outerWeightGradient: DenseVector[Double]) {

  def *(scalar: Double) = WeightGradient(scalar * innerWeightGradient,
    scalar * innerBiasGradient, scalar * outerWeightGradient)

  def :*=(scalar: Double): WeightGradient = {
    innerWeightGradient :*= scalar
    innerBiasGradient :*= scalar
    outerWeightGradient :*= scalar
    this
  }

  def /(scalar: Double) = WeightGradient(innerWeightGradient/scalar,
    innerBiasGradient/scalar, outerWeightGradient/scalar)

  def :/=(scalar: Double): WeightGradient = {
    innerWeightGradient :/= scalar
    innerBiasGradient :/= scalar
    outerWeightGradient :/= scalar
    this
  }

  def +(other: WeightGradient): WeightGradient =
    WeightGradient(innerWeightGradient + other.innerWeightGradient,
      innerBiasGradient + other.innerBiasGradient,
      outerWeightGradient + other.outerWeightGradient)

  def :+=(other: WeightGradient): WeightGradient = {
    innerWeightGradient :+= other.innerWeightGradient
    innerBiasGradient :+= other.innerBiasGradient
    outerWeightGradient :+= other.outerWeightGradient
    this
  }

}

