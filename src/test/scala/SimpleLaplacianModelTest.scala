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

    val model = SimpleLaplacianModel(net, 0d)
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

    val model = SimpleLaplacianModel(net, 0d)
    val result = DenseVector(20.248680491572, 20.21034960445042).t

    model.laplacian(x1) should be (result(0))
    model.laplacian(x2) should be (result(1))
    model.laplacianBatch(x) should be (result)

  }

  it should "calculate costs correctly" in {
    val w = DenseMatrix((1d,.5),(.25, .75))
    val b = DenseVector(.6,.3)
    val v = DenseVector(1d, .25)

    val net = SimpleNetwork(w,b,v)
    val x1 = DenseVector(2d,3d)
    val x2 = DenseVector(3d,2d)
    val x = DenseMatrix(x1,x2).t

    val model = SimpleLaplacianModel(net, 0d)

    model.cost(x2) should be (204.22911556705463)
    model.cost(x1) should be (205.00453082488423)
    model.batchCost(x) should be (sum(DenseVector(205.00453082488423,204.22911556705463 )))
  }

  it should "calculate weight gradients correctly" in {
    val w = DenseMatrix((1d,.5),(.25, .75))
    val b = DenseVector(.6,.3)
    val v = DenseVector(1d, .25)

    val net = SimpleNetwork(w,b,v)
    val x1 = DenseVector(2d,3d)
    val x2 = DenseVector(3d,4d)
    val x = DenseMatrix(x1,x2).t

    val model = SimpleLaplacianModel(net, 0d)
    val grad1 = model.costGradient(x1)
    val grad2 = model.costGradient(x2)
    val grad = model.costGradientBatch(x)

    val result1 = WeightGradient(DenseMatrix((-5.554195001804619, -11.83262019637254),
      (6.55599755831139,0.6142978287741843)),
      DenseVector(-4.852483926822637, -0.05951938292162778),
      DenseVector(328.92243589032734, 324.34650303776397))

    val result2 = WeightGradient(DenseMatrix((-22.11140866931294, -22.33019279627717),
      (8.585779193622825,-14.020532509445735)),
      DenseVector(-6.10216310391092, -2.5401184351795036),
      DenseVector(1632.8426928855063, 1636.605201581379))

    grad1 should be (result1)
    grad2 should be (result2)

    grad should be (grad1 + grad2)

  }
}
