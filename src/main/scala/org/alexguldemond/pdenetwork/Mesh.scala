package org.alexguldemond.pdenetwork

import breeze.linalg.DenseMatrix

trait Mesh {
  def iterator(batchSize: Int) : MeshIterator
}

trait MeshIterator {
  def nextBatch: DenseMatrix[Double]

  def hasNext: Boolean
}