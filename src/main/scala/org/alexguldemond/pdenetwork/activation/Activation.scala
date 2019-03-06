package org.alexguldemond.pdenetwork.activation

import breeze.linalg.{DenseMatrix, DenseVector}

trait Activation {
  def apply(x: Double): Double

  def apply(x: DenseVector[Double]) : DenseVector[Double]

  def apply(x: DenseMatrix[Double]): DenseMatrix[Double]

  def derivative(n: Int, x : Double): Double

  def derivative(n: Int, x: DenseVector[Double]): DenseVector[Double]

  def derivative(n: Int, x: DenseMatrix[Double]): DenseMatrix[Double]
}
