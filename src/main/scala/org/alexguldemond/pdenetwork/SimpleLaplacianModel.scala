package org.alexguldemond.pdenetwork

import breeze.linalg._

abstract class SimpleLaplacianModel(simpleNetwork: SimpleNetwork) extends SimpleModel {

  import SimpleLaplacianModel._

  override def derivativeMap: Map[MultiIndex, NetworkDerivative] =
      Map(zeroIndex -> simpleNetwork,
        x -> simpleNetwork.inputDerivative(x),
        y -> simpleNetwork.inputDerivative(y),
        xx -> simpleNetwork.inputDerivative(xx),
        yy -> simpleNetwork.inputDerivative(yy))

  override def operatorCoefficients(input: DenseMatrix[Double]): MultiIndexCoefficiants = {
      val x1 = input(0,::)
      val x2 = input(1,::)

      val x1minus1 = x1 - 1d
      val x2minus1 = x2 - 1d
      val x1x1Minus1 = x1 *:* x1minus1
      val x2x2Minus1 = x2 *:* x2minus1

      val constantTerm = diffOpBcSatisfier(input)

      MultiIndexCoefficiants(constantTerm,
        Map(zeroIndex -> (x1x1Minus1 + x2x2Minus1)*2d,
            x -> ((x1minus1 *:* x2x2Minus1 + x1 *:* x2x2Minus1)*2d),
            y -> ((x1x1Minus1 *:* x2minus1 + x1x1Minus1 *:* x2)*2d),
            xx -> (x1x1Minus1 *:* x2x2Minus1),
            yy -> (x1x1Minus1 *:* x2x2Minus1)))
  }

  override def apply(input: DenseVector[Double]): Double = {
    val x1 = input(0)
    val x2 = input(1)
    bcSatisfier(input) + x1 * (x1 - 1d) * x2 * (x2 - 1d) * simpleNetwork(input)
  }

  override def updateWeights(weightGradient: WeightGradient): Unit = {
    simpleNetwork.updateWeights(weightGradient)
  }

  def bcSatisfier(input: DenseVector[Double]): Double

  def diffOpBcSatisfier(input: DenseMatrix[Double]): Transpose[DenseVector[Double]]

}


object SimpleLaplacianModel {
  lazy val zeroIndex = MultiIndex(Array(0,0))
  lazy val x = MultiIndex(Array(1,0))
  lazy val y = MultiIndex(Array(0,1))
  lazy val xx = MultiIndex(Array(2,0))
  lazy val yy = MultiIndex(Array(0,2))

}