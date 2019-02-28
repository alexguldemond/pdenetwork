import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.numerics.sin
import breeze.numerics.constants.Pi
import org.alexguldemond.pdenetwork.{SimpleLaplacianModel, SimpleNetwork, WeightGradient}
import org.scalatest.{FlatSpec, Matchers}

class SimpleLaplacianModelTest extends FlatSpec with Matchers {

  "A SimpleLaplacianModel" should "apply correctly" in {
    val w = DenseMatrix((1d,.5),(.25, .75))
    val b = DenseVector(.6,.3)
    val v = DenseVector(1d, .25)

    val net = SimpleNetwork(w,b,v)
    val x = DenseVector(2d,3d)

    val result = 3d * sin(Pi * 2d) + 2d*(2d - 1d)*3d*(3d - 1d) * net(x)

    val model = SimpleLaplacianModel(net)
    model(x) should be (result)

  }

  it should "calculate laplacian correctly" in {
    val w = DenseMatrix((1d,.5),(.25, .75))
    val b = DenseVector(.6,.3)
    val v = DenseVector(1d, .25)

    val net = SimpleNetwork(w,b,v)
    val x1 = DenseVector(2d,3d)
    val x2 = DenseVector(3d,2d)
    val x = DenseMatrix(x1,x2).t

    val model = SimpleLaplacianModel(net)
    val result = DenseVector(20.248680491572003, 20.21034960445042).t

    model.diffOp(x1) should be (result(0))
    model.diffOp(x2) should be (result(1))
    model.diffOpBatch(x) should be (result)

  }

  it should "calculate costs correctly" in {
    val w = DenseMatrix((1d,.5),(.25, .75))
    val b = DenseVector(.6,.3)
    val v = DenseVector(1d, .25)

    val net = SimpleNetwork(w,b,v)
    val x1 = DenseVector(2d,3d)
    val x2 = DenseVector(3d,2d)
    val x = DenseMatrix(x1,x2).t

    val model = SimpleLaplacianModel(net)

    model.cost(x2) should be (204.22911556705463)
    model.cost(x1) should be (205.0045308248843)
    model.batchCost(x) should be (sum(DenseVector(205.0045308248843,204.22911556705463 )))
  }

  it should "calculate weight gradients correctly" in {
    val w = DenseMatrix((1d,.5),(.25, .75))
    val b = DenseVector(.6,.3)
    val v = DenseVector(1d, .25)

    val net = SimpleNetwork(w,b,v)
    val x1 = DenseVector(2d,3d)
    val x2 = DenseVector(3d,4d)
    val x = DenseMatrix(x1,x2).t

    val model = SimpleLaplacianModel(net)
    val grad1 = model.costGradient(x1)
    val grad2 = model.costGradient(x2)
    val grad = model.costGradientBatch(x)

    val result1 = WeightGradient(DenseMatrix((-5.5541950018046204, -11.83262019637254),
      (6.55599755831139,0.6142978287741843)),
      DenseVector(-4.852483926822638, -0.05951938292162779),
      DenseVector(328.9224358903274, 324.346503037764))

    grad1 should be (result1)

  }
}
