import breeze.linalg.{DenseMatrix, DenseVector}
import org.scalatest.{FlatSpec, Matchers}

class Test extends FlatSpec with Matchers{

  "Matrix" should "work" in {
    val vec = DenseVector(1d,2d)
    println(DenseMatrix(vec, vec).t)
  }

}
