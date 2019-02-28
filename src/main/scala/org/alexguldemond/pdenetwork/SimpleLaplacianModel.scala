package org.alexguldemond.pdenetwork

import breeze.linalg._
import breeze.numerics.constants.Pi
import breeze.numerics.sin

case class SimpleLaplacianModel(simpleNetwork: SimpleNetwork) extends SimpleModel {

  import SimpleLaplacianModel._

  override def data(input: DenseMatrix[Double]): Transpose[DenseVector[Double]] = DenseVector.zeros[Double](input.cols).t

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

      MultiIndexCoefficiants(x2 *:* sin(x1*Pi)*minusPiSquared*4d,
        Map(zeroIndex -> (x1x1Minus1 + x2x2Minus1)*2d,
            x -> ((x1minus1 *:* x2x2Minus1 + x1 *:* x2x2Minus1)*2d),
            y -> ((x1x1Minus1 *:* x2minus1 + x1x1Minus1 *:* x2)*2d),
            xx -> (x1x1Minus1 *:* x2x2Minus1),
            yy -> (x1x1Minus1 *:* x2x2Minus1)))
    }


  override def apply(input: DenseVector[Double]): Double = {
    val x1 = input(0)
    val x2 = input(1)
    x2*sin(4d*Pi * x1) + x1 * (1d - x1) * x2 * (1 - x2) * simpleNetwork(input)
  }

  override def updateWeights(weightGradient: WeightGradient): Unit = {
    simpleNetwork.updateWeights(weightGradient)
  }

}


object SimpleLaplacianModel {
  lazy val zeroIndex = MultiIndex(Array(0,0))
  lazy val x = MultiIndex(Array(1,0))
  lazy val y = MultiIndex(Array(0,1))
  lazy val xx = MultiIndex(Array(2,0))
  lazy val yy = MultiIndex(Array(0,2))

  lazy val minusPiSquared = -Pi*Pi

  def randomModel( hiddenLayerSize: Int): SimpleLaplacianModel =
    SimpleLaplacianModel(SimpleNetwork.randomNetwork(2, hiddenLayerSize))

}