package org.alexguldemond.pdenetwork
import breeze.linalg._
import breeze.numerics.sin
import breeze.numerics.constants.Pi

case class SimpleLaplacianModel(simpleNetwork: SimpleNetwork, learningRate: Double) extends Model {
  import SimpleLaplacianModel._

  override def cost(input: DenseVector[Double]): Double = {
    val l = laplacian(input)
    l*l/2d
  }

  override def batchCost(input: DenseMatrix[Double]): Double = {
    val l = laplacianBatch(input)
   sum((l *:* l)/2d)
  }

  def laplacian(input: DenseVector[Double]): Double = {
    val n = simpleNetwork(input)
    val nx = simpleNetwork.inputDerivative(x)(input)
    val ny = simpleNetwork.inputDerivative(y)(input)
    val nxx = simpleNetwork.inputDerivative(xx)(input)
    val nyy = simpleNetwork.inputDerivative(yy)(input)
    val x1 = input(0)
    val x2 = input(1)

    val x1minus1 = x1 - 1
    val x2minus1 = x2 - 1
    val x1x1Minus1 = x1*x1minus1
    val x2x2Minus1 = x2*x2minus1

    minusPiSquared*x2*sin(Pi*x1) +
      2*(x1x1Minus1 + x2x2Minus1)*n +
      2*(x1minus1 * x2x2Minus1 + x1 * x2x2Minus1) * nx +
      2*(x1x1Minus1 * x2minus1 + x1x1Minus1 * x2) * ny +
      (x1x1Minus1 * x2x2Minus1) * (nxx + nyy)
  }

  def laplacianBatch(input: DenseMatrix[Double]): Transpose[DenseVector[Double]] = {
    val n = simpleNetwork.applyBatch(input)
    val nx = simpleNetwork.inputDerivative(x).applyBatch(input)
    val ny = simpleNetwork.inputDerivative(y).applyBatch(input)
    val nxx = simpleNetwork.inputDerivative(xx).applyBatch(input)
    val nyy = simpleNetwork.inputDerivative(yy).applyBatch(input)
    val x1 = input(0,::)
    val x2 = input(1,::)

    val x1minus1 = x1 - 1d
    val x2minus1 = x2 - 1d
    val x1x1Minus1 = x1 *:* x1minus1
    val x2x2Minus1 = x2 *:* x2minus1

    x2 *:* sin(x1*Pi)*minusPiSquared +
      (x1x1Minus1 + x2x2Minus1)*2d *:* n +
      ((x1minus1 *:* x2x2Minus1 + x1 *:* x2x2Minus1)*2d) *:* nx +
      ((x1x1Minus1 *:* x2minus1 + x1x1Minus1 *:* x2)*2d) *:* ny +
      (x1x1Minus1 *:* x2x2Minus1) *:* (nxx + nyy)
  }

  def costGradient(input: DenseVector[Double]): WeightGradient = {
    val n = simpleNetwork(input)

    val dnx = simpleNetwork.inputDerivative(x)
    val dny = simpleNetwork.inputDerivative(y)
    val dnxx = simpleNetwork.inputDerivative(xx)
    val dnyy = simpleNetwork.inputDerivative(yy)

    val nx = dnx(input)
    val ny = dny(input)
    val nxx = dnxx(input)
    val nyy = dnyy(input)
    val x1 = input(0)
    val x2 = input(1)

    val x1minus1 = x1 - 1
    val x2minus1 = x2 - 1
    val x1x1Minus1 = x1*x1minus1
    val x2x2Minus1 = x2*x2minus1

    val laplacian: Double = minusPiSquared*x2*sin(Pi*x1) +
      2*(x1x1Minus1 + x2x2Minus1)*n +
      2*(x1minus1 * x2x2Minus1 + x1 * x2x2Minus1) * nx +
      2*(x1x1Minus1 * x2minus1 + x1x1Minus1 * x2) * ny +
      (x1x1Minus1 * x2x2Minus1) * (nxx + nyy)

    val nw = simpleNetwork.weightGradient(input)
    val nxw = dnx.weightGradient(input)
    val nyw = dny.weightGradient(input)
    val nxxw = dnxx.weightGradient(input)
    val nyyw = dnyy.weightGradient(input)

    val weightGradient:WeightGradient = (nw * 2 * (x1x1Minus1 + x2x2Minus1)) +
      nxw * 2 * (x1minus1 * x2x2Minus1 + x1 * x2x2Minus1) +
      nyw * 2 * (x1x1Minus1 * x2minus1 + x1x1Minus1 * x2) +
      (nxxw + nyyw) * (x1x1Minus1 * x2x2Minus1)

    weightGradient :*= laplacian
  }

  override def costGradientBatch(input: DenseMatrix[Double]): WeightGradient = {
    val n = simpleNetwork.applyBatch(input)

    val dnx = simpleNetwork.inputDerivative(x)
    val dny = simpleNetwork.inputDerivative(y)
    val dnxx = simpleNetwork.inputDerivative(xx)
    val dnyy = simpleNetwork.inputDerivative(yy)

    val nx = dnx.applyBatch(input)
    val ny = dny.applyBatch(input)
    val nxx = dnxx.applyBatch(input)
    val nyy = dnyy.applyBatch(input)
    val x1 = input(0,::)
    val x2 = input(1,::)

    val x1minus1 = x1 - 1d
    val x2minus1 = x2 - 1d
    val x1x1Minus1 = x1 *:* x1minus1
    val x2x2Minus1 = x2 *:* x2minus1

    val laplacian = x2 *:* sin(x1*Pi)*minusPiSquared +
      (x1x1Minus1 + x2x2Minus1)*2d *:* n +
      ((x1minus1 *:* x2x2Minus1 + x1 *:* x2x2Minus1)*2d) *:* nx +
      ((x1x1Minus1 *:* x2minus1 + x1x1Minus1 *:* x2)*2d) *:* ny +
      (x1x1Minus1 *:* x2x2Minus1) *:* (nxx + nyy)

    val nw = simpleNetwork.weightGradientBatch(input)
    val nxw = dnx.weightGradientBatch(input)
    val nyw = dny.weightGradientBatch(input)
    val nxxw = dnxx.weightGradientBatch(input)
    val nyyw = dnyy.weightGradientBatch(input)

    val weightGradient: WeightGradientBatch = nw.elemProd((x1x1Minus1 + x2x2Minus1)*2d)
    weightGradient.inPlaceSum( nxw.elemProd( (x1minus1 *:* x2x2Minus1 + x1 *:* x2x2Minus1) * 2d) )
    weightGradient.inPlaceSum( nyw.elemProd( (x1x1Minus1 *:* x2minus1 + x1x1Minus1 *:* x2) * 2d) )
    weightGradient.inPlaceSum( (nxxw sum nyyw).elemProd(x1x1Minus1 *:* x2x2Minus1) )

    weightGradient.dotSum(laplacian)
  }


  override def apply(input: DenseVector[Double]): Double = {
    val x1 = input(0)
    val x2 = input(1)
    x2*sin(Pi * x1) + x1 * (1d - x1) * x2 * (1 - x2) * simpleNetwork(input)
  }

  override def update(input: DenseMatrix[Double]): Unit = {
    val averageGradient = costGradientBatch(input) * (-1d*learningRate/input.cols)
    simpleNetwork.updateWeights(averageGradient)
  }

  override def fit(iter: MeshIterator): Unit = {
    while(iter.hasNext) {
      val batch = iter.nextBatch
      update(batch)
    }
  }
}

object SimpleLaplacianModel {
  lazy val x = MultiIndex(Array(1,0))
  lazy val y = MultiIndex(Array(0,1))
  lazy val xx = MultiIndex(Array(2,0))
  lazy val yy = MultiIndex(Array(0,2))

  lazy val minusPiSquared = -Pi*Pi

  def randomModel(learningRate: Double, hiddenLayerSize: Int): SimpleLaplacianModel =
    SimpleLaplacianModel(SimpleNetwork.randomNetwork(2, hiddenLayerSize), learningRate)

}