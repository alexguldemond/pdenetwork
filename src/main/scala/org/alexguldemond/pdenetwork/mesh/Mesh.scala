package org.alexguldemond.pdenetwork.mesh

import breeze.linalg.DenseMatrix

trait Mesh {
  def iterator(batchSize: Int) : MeshIterator

  def numberOfPoints: Int
}

trait MeshIterator {
  def nextBatch: DenseMatrix[Double]

  def hasNext: Boolean
}