import org.alexguldemond.pdenetwork.mesh.Uniform2DMesh
import org.scalatest.{FlatSpec, Matchers}

class Uniform2DMeshTest extends FlatSpec with Matchers {

  "A Uniform2DMesh" should "work" in {

    val mesh = Uniform2DMesh(0.02)
    mesh.numberOfPoints should be (50)

  }
}
