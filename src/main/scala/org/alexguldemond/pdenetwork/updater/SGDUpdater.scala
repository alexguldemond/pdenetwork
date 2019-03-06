package org.alexguldemond.pdenetwork.updater

import breeze.linalg.DenseMatrix
import org.alexguldemond.pdenetwork.model.Model

case class SGDUpdater(learningRate: Double) extends StochasticUpdater {
  override def updateModel(model: Model, input: DenseMatrix[Double]): Model = {
    val avg = model.averageGradient(input) * -learningRate
    model.updateWeights(avg)
    model
  }
}
