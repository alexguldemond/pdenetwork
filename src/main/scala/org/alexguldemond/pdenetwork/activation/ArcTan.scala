package org.alexguldemond.pdenetwork.activation

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{tanh}
import ArcTanDerivatives._

object ArcTan extends Activation {
  override def apply(x: Double): Double = tanh(x)

  override def apply(x: DenseVector[Double]): DenseVector[Double] = tanh(x)

  override def apply(x: DenseMatrix[Double]): DenseMatrix[Double] = tanh(x)

  override def derivative(n: Int, x: Double): Double = n match {
    case 0 => tanh(x)
    case 1 => tanhFirstDerivative(x)
    case 2 => tanhSecondDerivative(x)
    case 3 => tanhThirdDerivative(x)
    case 4 => tanhFourthDerivative(x)
    case _ => throw new IllegalArgumentException("Higher derivatives not implemented")
  }

  override def derivative(n: Int, x: DenseVector[Double]): DenseVector[Double] = n match {
    case 0 => tanh(x)
    case 1 => tanhFirstDerivative(x)
    case 2 => tanhSecondDerivative(x)
    case 3 => tanhThirdDerivative(x)
    case 4 => tanhFourthDerivative(x)
    case _ => throw new IllegalArgumentException("Higher derivatives not implemented")
  }

  override def derivative(n: Int, x: DenseMatrix[Double]): DenseMatrix[Double] = n match {
    case 0 => tanh(x)
    case 1 => tanhFirstDerivative(x)
    case 2 => tanhSecondDerivative(x)
    case 3 => tanhThirdDerivative(x)
    case 4 => tanhFourthDerivative(x)
    case _ => throw new IllegalArgumentException("Higher derivatives not implemented")
  }
}
