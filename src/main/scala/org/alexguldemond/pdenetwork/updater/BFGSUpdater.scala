package org.alexguldemond.pdenetwork.updater
import breeze.linalg.DenseMatrix
import org.alexguldemond.pdenetwork.model.Model

case class BFGSUpdater(eta0: Double, tau: Double ,lambda: Double, c: Double = .1, initialHessian: Double = 1.0E-8) extends Updater {

  private var timeStep = 0

  private var hessianGuess = DenseMatrix((0d))

  private def etaT = tau * eta0 / (timeStep + tau)

  override def updateModel(model: Model, input: DenseMatrix[Double]): Model = {
    val grad = model.averageGradient(input).toDenseVector
    val weights = model.weightVector.toDenseVector
    val identity = DenseMatrix.eye[Double](grad.length)
    if (timeStep == 0) {
      hessianGuess = initialHessian * identity
    }

    val direction = -hessianGuess * grad
    val scaledDir = (etaT/c) * direction

    //Update happens here
    model.updateWeights(scaledDir)

    //Now we need to update our approximation of hessian
    val diff = model.averageGradient(input).toDenseVector - grad + lambda*scaledDir
    if (timeStep == 0) {
      hessianGuess := (scaledDir dot diff)/(diff dot diff) * DenseMatrix.eye[Double](grad.length)
    }
    val hessianUpdate = 1d/(scaledDir dot diff)
    hessianGuess :=
      (identity - hessianUpdate * scaledDir * diff.t) * hessianGuess * (identity - hessianUpdate * diff * scaledDir.t)
    hessianGuess :+= c * hessianUpdate * scaledDir * scaledDir.t
    timeStep = timeStep + 1

    model
  }
}
