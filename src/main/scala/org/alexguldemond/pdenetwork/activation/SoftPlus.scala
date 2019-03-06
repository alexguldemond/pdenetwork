package org.alexguldemond.pdenetwork.activation
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sigmoid
import org.alexguldemond.pdenetwork.activation.SigmoidDerivatives._

object SoftPlus extends Activation {
  override def apply(x: Double): Double = SigmoidDerivatives.softPlus(x)

  override def apply(x: DenseVector[Double]): DenseVector[Double] = softPlus(x)

  override def apply(x: DenseMatrix[Double]): DenseMatrix[Double] = softPlus(x)

  override def derivative(n: Int, x: Double): Double = n match {
    case 0 => softPlus(x)
    case 1 => sigmoid(x)
    case 2 => sigmoidFirstDerivative(x)
    case 3 => sigmoidSecondDerivative(x)
    case 4 => sigmoidThirdDerivative(x)
    case 5 => sigmoidFourthDerivative(x)
    case _ => throw new IllegalArgumentException("Higher derivatives not implemented")
  }

  override def derivative(n: Int, x: DenseVector[Double]): DenseVector[Double] = n match {
    case 0 => softPlus(x)
    case 1 => sigmoid(x)
    case 2 => sigmoidFirstDerivative(x)
    case 3 => sigmoidSecondDerivative(x)
    case 4 => sigmoidThirdDerivative(x)
    case 5 => sigmoidFourthDerivative(x)
    case _ => throw new IllegalArgumentException("Higher derivatives not implemented")
  }

  override def derivative(n: Int, x: DenseMatrix[Double]): DenseMatrix[Double] = n match {
    case 0 => softPlus(x)
    case 1 => sigmoid(x)
    case 2 => sigmoidFirstDerivative(x)
    case 3 => sigmoidSecondDerivative(x)
    case 4 => sigmoidThirdDerivative(x)
    case 5 => sigmoidFourthDerivative(x)
    case _ => throw new IllegalArgumentException("Higher derivatives not implemented")
  }
}
