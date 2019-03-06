package org.alexguldemond.pdenetwork.activation
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sigmoid
import org.alexguldemond.pdenetwork.activation.SigmoidDerivatives.{sigmoidFirstDerivative, sigmoidFourthDerivative, sigmoidSecondDerivative, sigmoidThirdDerivative}

object Sigmoid extends Activation {
  override def apply(x: Double): Double = sigmoid(x)

  override def apply(x: DenseVector[Double]): DenseVector[Double] = sigmoid(x)

  override def apply(x: DenseMatrix[Double]): DenseMatrix[Double] = sigmoid(x)

  override def derivative(n: Int, x: Double): Double = n match {
    case 0 => sigmoid(x)
    case 1 => sigmoidFirstDerivative(x)
    case 2 => sigmoidSecondDerivative(x)
    case 3 => sigmoidThirdDerivative(x)
    case 4 => sigmoidFourthDerivative(x)
    case _ => throw new IllegalArgumentException("Higher derivatives not implemented")
  }

  override def derivative(n: Int, x: DenseVector[Double]): DenseVector[Double] = n match {
    case 0 => sigmoid(x)
    case 1 => sigmoidFirstDerivative(x)
    case 2 => sigmoidSecondDerivative(x)
    case 3 => sigmoidThirdDerivative(x)
    case 4 => sigmoidFourthDerivative(x)
    case _ => throw new IllegalArgumentException("Higher derivatives not implemented")
  }

  override def derivative(n: Int, x: DenseMatrix[Double]): DenseMatrix[Double] = n match {
    case 0 => sigmoid(x)
    case 1 => sigmoidFirstDerivative(x)
    case 2 => sigmoidSecondDerivative(x)
    case 3 => sigmoidThirdDerivative(x)
    case 4 => sigmoidFourthDerivative(x)
    case _ => throw new IllegalArgumentException("Higher derivatives not implemented")
  }
}
