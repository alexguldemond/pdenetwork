import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sigmoid
import org.alexguldemond.pdenetwork.activation.SigmoidDerivatives.sigmoidFirstDerivative
import org.alexguldemond.pdenetwork.network.{SimpleNetwork, WeightGradientBatch, WeightVector}
import org.scalatest.{FlatSpec, Matchers}

class SimpleNetworkTest extends FlatSpec with Matchers {

  "A simple network" should "compute preoutputs correctly" in {
    val w = DenseMatrix((1d,.5),(.25, .75))
    val b = DenseVector(.6,.3)
    val v = DenseVector(1d, .25)

    val net = SimpleNetwork(WeightVector(w,b,v))
    val x1 = DenseVector(1d,2d)
    val x2 = DenseVector(1d,.5)
    val x = DenseMatrix(x1,x2).t

    val z1 = w*x1 + b
    val z2 = w*x2 + b
    val z = DenseMatrix(z1,z2).t

    net.hiddenPreOutput(x1) should be (z1)
    net.hiddenPreOutput(x2) should be (z2)
    net.hiddenPreOutputBatch(x) should be (z)
  }

  it should "apply correctly" in {
    val w = DenseMatrix((1d,.5),(.25, .75))
    val b = DenseVector(.6,.3)
    val v = DenseVector(1d, .25)

    val net = SimpleNetwork(WeightVector(w,b,v))
    val x1 = DenseVector(1d,2d)
    val x2 = DenseVector(1d,.5)
    val x = DenseMatrix(x1,x2).t

    val z1 = w*x1 + b
    val z2 = w*x2 + b

    val a1 = v dot sigmoid(z1)
    val a2 = v dot sigmoid(z2)

    val a = DenseVector(a1, a2).t
    net(x1) should be (a1)
    net(x2) should be (a2)
    net.applyBatch(x) should be (a)

  }

  it should "compute weight gradients correctly" in {
    val w = DenseMatrix((1d,.5),(.25, .75))
    val b = DenseVector(.6,.3)
    val v = DenseVector(1d, .25)

    val net = SimpleNetwork(WeightVector(w,b,v))
    val x1 = DenseVector(1d,2d)
    val x2 = DenseVector(1d,.5)
    val x = DenseMatrix(x1,x2).t

    val x1Pre = net.hiddenPreOutput(x1)
    val x2Pre = net.hiddenPreOutput(x2)

    val x1Grad = WeightVector((v *:* sigmoidFirstDerivative(x1Pre)) * x1.t,
      v *:* sigmoidFirstDerivative(x1Pre),
      sigmoid(x1Pre))

    val x2Grad = WeightVector((v *:* sigmoidFirstDerivative(x2Pre)) * x2.t,
      v *:* sigmoidFirstDerivative(x2Pre),
      sigmoid(x2Pre))

    val xGrad = WeightGradientBatch( Seq(x1Grad.innerWeights, x2Grad.innerWeights),
      DenseMatrix(x1Grad.innerBias, x2Grad.innerBias).t,
      DenseMatrix(x1Grad.outerWeight, x2Grad.outerWeight).t)

    net.weightGradient(x1) should be (x1Grad)
    net.weightGradient(x2) should be (x2Grad)
    net.weightGradientBatch(x) should be (xGrad)

  }

}
