import breeze.linalg._
import org.alexguldemond.pdenetwork.activation.SigmoidDerivatives.sigmoidThirdDerivative
import org.alexguldemond.pdenetwork.network.{MultiIndex, SimpleDerivative, SimpleNetwork, WeightVector}
import org.scalatest.{FlatSpec, Matchers}

class SimpleDerivativeTest extends FlatSpec with Matchers{

  "matrixVectorPowerProduct" should "return the correct result" in {
    val mat = DenseMatrix((1d, 2d), (3d, 4d))
    val powers = DenseVector(1d,1d)

    val result = DenseVector(2d, 12d)
    SimpleDerivative.matrixVectorPowerProduct(mat, powers) should be (result)

    val powers2 = DenseVector(3d, 2d)
    val result2 = DenseVector(4d, 432.0)
    SimpleDerivative.matrixVectorPowerProduct(mat, powers2) should be (result2)

  }

  "A SimpleDerivative" should "apply correctly" in {
    val innerWeights = DenseMatrix((1.0, .75),(.5, .25))
    val innerBias = DenseVector(.5, 1.0)
    val outerWeights = DenseVector(1.0, .5)

    val simpleNetwork: SimpleNetwork = SimpleNetwork(WeightVector(innerWeights, innerBias, outerWeights))
    val multiIndex: MultiIndex = MultiIndex(Array(2,1))

    val simpleDerivative = SimpleDerivative(simpleNetwork, multiIndex)
    simpleDerivative.weightModifier should be (DenseVector(0.75, 0.0625))

    simpleDerivative.modifiedOuterWeight should be (DenseVector(0.75, 0.03125))

    simpleDerivative.innerWeightGradMod should be (DenseMatrix((.75, 1.0), (0.125, 0.25)))

    val x = DenseVector(.5d, .5d)
    val result = innerWeights * x + innerBias
    val sigma = sigmoidThirdDerivative(result)
    val modOuter = outerWeights *:* SimpleDerivative.matrixVectorPowerProduct(innerWeights, multiIndex.asVector)
    val finalResult = modOuter dot sigma

    simpleDerivative(x) should be (finalResult)
  }

  "A SimpleDerivative" should "construct the weight gradients correctly" in {
    val innerWeights = DenseMatrix((1.0, .75),(.5, .25))
    val innerBias = DenseVector(.5, 1.0)
    val outerWeights = DenseVector(1.0, .5)
    val x = DenseVector(.5d, .5d)

    val simpleNetwork: SimpleNetwork = SimpleNetwork(WeightVector(innerWeights, innerBias, outerWeights))
    val multiIndex: MultiIndex = MultiIndex(Array(2,1))

    val simpleDerivative = SimpleDerivative(simpleNetwork, multiIndex)
    val grad = simpleDerivative.weightGradient(x)
    grad.outerWeight should be (DenseVector(0.004046300496763141, 3.371917080635951E-4))
    grad.innerBias should be (DenseVector(0.06722375388622398, 0.0028009897452593324))

    grad.innerWeights should be (DenseMatrix((0.04170447793663827,0.039006944272129515),
                                                    (0.0020748782887568566, 0.0020748782887568566)))
  }

  "A SimpleDerivative" should "not divide be zero when constructing gradients" in {
    val innerWeights = DenseMatrix((0d, 0d),(0d, 0d))
    val innerBias = DenseVector(.5, 1.0)
    val outerWeights = DenseVector(1.0, .5)
    val x = DenseVector(.5d, .5d)

    val simpleNetwork: SimpleNetwork = SimpleNetwork(WeightVector(innerWeights, innerBias, outerWeights))
    val multiIndex: MultiIndex = MultiIndex(Array(2,0))
    val simpleDerivative = SimpleDerivative(simpleNetwork, multiIndex)
    val grad = simpleDerivative.weightGradient(x)
    val result = grad.innerWeights(::,1)
    result should be (DenseVector(0.0, 0.0))
  }

  "A SimpleDerivative" should "construct the weight gradients correctly for batches" in {
    val innerWeights = DenseMatrix((1.0, .75),(.5, .25))
    val innerBias = DenseVector(.5, 1.0)
    val outerWeights = DenseVector(1.0, .5)
    val x = DenseMatrix((.5,.75), (.5, .25))
    val x1 = DenseVector(.5,.5)
    val x2 = DenseVector(.75,.25)

    val simpleNetwork: SimpleNetwork = SimpleNetwork(WeightVector(innerWeights, innerBias, outerWeights))
    val multiIndex: MultiIndex = MultiIndex(Array(2,1))

    val simpleDerivative = SimpleDerivative(simpleNetwork, multiIndex)

    val grad = simpleDerivative.weightGradientBatch(x)
    val grad1 = simpleDerivative.weightGradient(x1)
    val grad2 = simpleDerivative.weightGradient(x2)

    grad.outerWeightGradients should be (DenseMatrix(grad1.outerWeight, grad2.outerWeight).t)
    grad.innerBiasGradients should be (DenseMatrix(grad1.innerBias, grad2.innerBias).t)
    grad.innerWeightGradients should be (Seq(grad1.innerWeights, grad2.innerWeights))

  }
}
