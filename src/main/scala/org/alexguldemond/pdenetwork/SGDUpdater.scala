package org.alexguldemond.pdenetwork
import breeze.linalg.DenseMatrix

case class SGDUpdater(learningRate: Double) extends Updater {
  override def updateModel(model: Model, input: DenseMatrix[Double]): Model = {
    val avg = model.averageGradient(input) * -learningRate
    model.updateWeights(avg)
    model
  }
}
