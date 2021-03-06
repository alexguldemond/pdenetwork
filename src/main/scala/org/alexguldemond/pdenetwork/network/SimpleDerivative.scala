package org.alexguldemond.pdenetwork.network

import breeze.linalg._

case class SimpleDerivative(simpleNetwork: SimpleNetwork, multiIndex: MultiIndex)
  extends NetworkDerivative {

  import simpleNetwork.{hiddenPreOutput, hiddenPreOutputBatch}

  val weightModifier: DenseVector[Double] = SimpleDerivative.matrixVectorPowerProduct(W, multiIndex.asVector)

  val modifiedOuterWeight: DenseVector[Double] = simpleNetwork.outerWeights *:* weightModifier

  private[this] def W: DenseMatrix[Double] = simpleNetwork.innerWeights

  private[this] def activation = simpleNetwork.activation

  val innerWeightGradMod: DenseMatrix[Double] = {
    val mat = DenseMatrix.zeros[Double](W.rows, W.cols)
    for (i: Int <- 0 to W.cols - 1) {
      val powers = multiIndex.asVector
      powers(i) = if (powers(i) == 0d) {
        powers(i)
      } else {
        powers(i) - 1
      }
      mat(::, i) := SimpleDerivative.matrixVectorPowerProduct(W, powers)
    }
    mat
  }

  override def apply(input: DenseVector[Double]): Double =
    modifiedOuterWeight dot activation.derivative(multiIndex.total, hiddenPreOutput(input))

  override def applyBatch(input: DenseMatrix[Double]): Transpose[DenseVector[Double]] =
    modifiedOuterWeight.t * activation.derivative(multiIndex.total, hiddenPreOutputBatch(input))

  override def weightGradient(input: DenseVector[Double]) : WeightVector = {
    val preOutput = hiddenPreOutput(input)

    val sigma = activation.derivative(multiIndex.total, preOutput)

    val outerWeightGrad = weightModifier *:* sigma
    val innerBiasGrad = modifiedOuterWeight *:* activation.derivative(multiIndex.total + 1, preOutput)

    val v = simpleNetwork.outerWeights

    val innerWeightGrad = innerBiasGrad * (input.t) + ((v *:* sigma) * multiIndex.asVector.t) *:* innerWeightGradMod

    WeightVector(innerWeightGrad, innerBiasGrad, outerWeightGrad)
  }

  override def weightGradientBatch(input: DenseMatrix[Double]): WeightGradientBatch = {

    val preOutput = hiddenPreOutputBatch(input)
    val sigma = activation.derivative(multiIndex.total, preOutput)

    val outerWeightGradMat = sigma(::, *) *:* weightModifier

    val innerBiasGradMat = activation.derivative(multiIndex.total + 1, preOutput)
    innerBiasGradMat := innerBiasGradMat(::, *) *:* modifiedOuterWeight

    val v = simpleNetwork.outerWeights
    val innerWeightMats: Seq[DenseMatrix[Double]] = Seq.tabulate(input.cols){ i =>
      innerBiasGradMat(::, i) * (input(::,i).t) + ((v *:* sigma(::,i)) * multiIndex.asVector.t) *:* innerWeightGradMod
    }

    WeightGradientBatch(innerWeightMats, innerBiasGradMat, outerWeightGradMat)
  }
}

object SimpleDerivative {

  /**
    * Given a matrix W and vector V, returns Hadamard Product of Wj to the Vj
    * @param matrix the base matrix
    * @param powers the vector of exponents
    * @return the result
    */
  def matrixVectorPowerProduct(matrix : DenseMatrix[Double], powers: DenseVector[Double]): DenseVector[Double] = {
    val powerMatrix = matrix(*,::) ^:^ powers
    product(powerMatrix(*,::))
  }

}