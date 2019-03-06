package org.alexguldemond.pdenetwork.updater

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{pow, sqrt}
import org.alexguldemond.pdenetwork.model.Model

case class AdamUpdater(learningRate: Double ,
                       moment1DecayRate: Double = 0.9,
                       moment2DecayRate: Double = 0.999,
                       epsilon: Double = 1.0E-8) extends StochasticUpdater{

  private var timeStep = 0

  //TODO: Change to a cats State
  private var moment1: DenseVector[Double] = DenseVector(0d)

  private var moment2: DenseVector[Double] = DenseVector(0d)

  override def updateModel(model: Model, input: DenseMatrix[Double]): Model = {
    val grad = model.averageGradient(input).toDenseVector

    if (timeStep == 0) {
      moment1 = DenseVector.zeros[Double](grad.length)
      moment2 = DenseVector.zeros[Double](grad.length)
    }

    timeStep = timeStep + 1
    moment1 := moment1DecayRate*moment1 + (1d - moment1DecayRate)* grad
    moment2 := moment2DecayRate*moment2 + (1d - moment2DecayRate) * (grad *:* grad)
    val correctedMoment1 = moment1 / (1 - pow(moment1DecayRate, timeStep))
    val correctedMoment2 = moment2 / (1 - pow(moment2DecayRate, timeStep))

    model.updateWeights( -learningRate * correctedMoment1 /:/ (sqrt(correctedMoment2) + epsilon) )
    model
  }


}
