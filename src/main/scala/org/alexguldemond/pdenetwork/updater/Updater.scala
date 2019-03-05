package org.alexguldemond.pdenetwork.updater

import breeze.linalg.DenseMatrix
import org.alexguldemond.pdenetwork.model.Model

trait Updater {

  def updateModel(model: Model, input: DenseMatrix[Double]): Model

}
