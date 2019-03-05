package org.alexguldemond.pdenetwork.network

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.{Gaussian => G}
import org.alexguldemond.pdenetwork.activation.SigmoidDerivatives._

case class SimpleNetwork(weightVector: WeightVector) extends Network {

  def innerWeights = weightVector.innerWeights

  def innerBias = weightVector.innerBias

  def outerWeights = weightVector.outerWeight

  override def apply(input: DenseVector[Double]): Double = outerWeights dot sigmoid(hiddenPreOutput(input))

  override def applyBatch(input: DenseMatrix[Double]): Transpose[DenseVector[Double]] =
    outerWeights.t * sigmoid(hiddenPreOutputBatch(input))

  override def weightGradient(input: DenseVector[Double]) : WeightVector = {
    val preOutput = hiddenPreOutput(input)

    val outerWeightGrad = sigmoid(preOutput)
    val innerBiasGrad = outerWeights *:* sigmoidFirstDerivative(preOutput)
    val innerWeightGrad = innerBiasGrad * input.t

    WeightVector(innerWeightGrad, innerBiasGrad, outerWeightGrad)
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

  override def updateWeights(weightGradient: WeightVector): Network = {
    innerWeights :+= weightGradient.innerWeights
    innerBias :+= weightGradient.innerBias
    outerWeights :+= weightGradient.outerWeight
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

  def randomNetwork(inputSize: Int, hiddenLayerSize: Int, testSize: Int = 1): SimpleNetwork = {
    val normal = G(0,1/sqrt(testSize))
    val w = DenseMatrix.rand(hiddenLayerSize, inputSize, normal)
    val v = DenseVector.rand(hiddenLayerSize, normal)
    val b = DenseVector.rand(hiddenLayerSize, normal)
    SimpleNetwork(WeightVector(w, b, v))
  }
}
