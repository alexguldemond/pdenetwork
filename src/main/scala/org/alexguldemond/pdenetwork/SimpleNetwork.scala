package org.alexguldemond.pdenetwork

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Gaussian
import org.alexguldemond.pdenetwork.SigmoidDerivatives._

case class SimpleNetwork(innerWeights: DenseMatrix[Double], innerBias: DenseVector[Double], outerWeights: DenseVector[Double]) extends Network {

  override def apply(input: DenseVector[Double]): Double = outerWeights dot sigmoid(hiddenPreOutput(input))

  override def applyBatch(input: DenseMatrix[Double]): Transpose[DenseVector[Double]] =
    outerWeights.t * sigmoid(hiddenPreOutputBatch(input))

  override def weightGradient(input: DenseVector[Double]) : WeightGradient = {
    val preOutput = hiddenPreOutput(input)

    val outerWeightGrad = sigmoid(preOutput)
    val innerBiasGrad = outerWeights *:* sigmoidFirstDerivative(preOutput)
    val innerWeightGrad = innerBiasGrad * input.t

    WeightGradient(innerWeightGrad, innerBiasGrad, outerWeightGrad)
  }

  override def weightGradientBatch(input: DenseMatrix[Double]): WeightGradientBatch = {
    val preOutput = hiddenPreOutputBatch(input)

    val outerWeightGrad: DenseMatrix[Double] = sigmoid(preOutput)

    val sigmoidFirst: DenseMatrix[Double] = sigmoidFirstDerivative(preOutput)
    val innerBiasGrad: DenseMatrix[Double] = sigmoidFirst(::, *) *:* outerWeights

    val innerWeightGrad: Seq[DenseMatrix[Double]] = Seq.tabulate(input.cols){ i =>
      innerBiasGrad(::, i) * (input(::,i).t)
    }

    WeightGradientBatch(innerWeightGrad, innerBiasGrad, outerWeightGrad)
  }

  def hiddenPreOutput(input: DenseVector[Double]): DenseVector[Double] = (innerWeights * input) + innerBias

  def hiddenPreOutputBatch(input: DenseMatrix[Double]): DenseMatrix[Double] = {
    val mat = innerWeights * input
    mat(::, *) + innerBias
  }

  override def inputDerivative(multiIndex: MultiIndex): NetworkDerivative = SimpleDerivative(this, multiIndex)

  override def updateWeights(weightGradient: WeightGradient): Network = {
    innerWeights :+= weightGradient.innerWeightGradient
    innerBias :+= weightGradient.innerBiasGradient
    outerWeights :+= weightGradient.outerWeightGradient
    this
  }
}

object SimpleNetwork {
  def getDerivative(i: Int, input: DenseVector[Double]): DenseVector[Double] = i match {
    case 0 => sigmoid(input)
    case 1 => sigmoidFirstDerivative(input)
    case 2 => sigmoidSecondDerivative(input)
    case 3 => sigmoidThirdDerivative(input)
    case 4 => sigmoidFourthDerivative(input)
    case _ => throw new IllegalArgumentException("Higher derivatives not implemented")
  }

  def getDerivative(i: Int, input: DenseMatrix[Double]): DenseMatrix[Double] = i match {
    case 0 => sigmoid(input)
    case 1 => sigmoidFirstDerivative(input)
    case 2 => sigmoidSecondDerivative(input)
    case 3 => sigmoidThirdDerivative(input)
    case 4 => sigmoidFourthDerivative(input)
    case _ => throw new IllegalArgumentException("Higher derivatives not implemented")
  }

  def randomNetwork(inputSize: Int, hiddenLayerSize: Int): SimpleNetwork = {
    val normal = Gaussian(0,1)
    val w = DenseMatrix.rand(hiddenLayerSize, inputSize, normal)
    val v = DenseVector.rand(hiddenLayerSize, normal)
    val b = DenseVector.rand(hiddenLayerSize, normal)
    SimpleNetwork(w, b, v)
  }
}
