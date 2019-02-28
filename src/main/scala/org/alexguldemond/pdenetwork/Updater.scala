package org.alexguldemond.pdenetwork

import breeze.linalg.DenseMatrix

trait Updater {

  def updateModel(model: Model, input: DenseMatrix[Double]): Model

}
