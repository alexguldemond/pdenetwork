package org.alexguldemond.pdenetwork.updater
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.optimize.{DiffFunction, LBFGS}
import org.alexguldemond.pdenetwork.model.Model

case class LBFGSUpdater(epochs: Int, historySize: Int) extends FullUpdater {

  /**
    *
    * @param model
    * @param input The full training set
    * @return
    */
  override def updateModel(model: Model, input: DenseMatrix[Double]): Model = {

    val lbfgs = new LBFGS[DenseVector[Double]](epochs, historySize)

    val f = new DiffFunction[DenseVector[Double]] {
      override def calculate(x: DenseVector[Double]): (Double, DenseVector[Double]) = {
        val tempModel = model.copyArchitecture(x)
        (tempModel.averageCost(input), tempModel.averageGradient(input).toDenseVector)

      }
    }
    
    val optimumWeights = lbfgs.minimize(f, model.weightVector.toDenseVector)
    model.copyArchitecture(optimumWeights)
  }
}
