import breeze.linalg.{DenseMatrix, DenseVector}
import org.alexguldemond.pdenetwork.network.WeightVector
import org.scalatest.{FlatSpec, Matchers}

class WeightGradientTest extends FlatSpec with Matchers{
  "A WeightGradient" should "convert back and forth correctly" in {
    val weightGrad = WeightVector(DenseMatrix((1d, 2d), (3d, 4d), (5d, 6d)),
      DenseVector(7d, 8d, 9d),
      DenseVector(10d, 11d, 8d))

    WeightVector.vecToWeightGrad(weightGrad.toDenseVector, 2, 3) should be (weightGrad)
  }

}
