package org.alexguldemond.pdenetwork.mesh

import breeze.linalg.{DenseMatrix, DenseVector}

import scala.util.Random

case class Uniform2DMesh(delta: Double) extends Mesh {

  override lazy val numberOfPoints = (1d/delta).floor.toInt

  lazy val data = List.tabulate(numberOfPoints * numberOfPoints){i => (i/numberOfPoints, i % numberOfPoints)}

  class Uniform2DMeshIterator(batchSize: Int) extends MeshIterator {

    lazy val iter = Random.shuffle(data).iterator

    override def nextBatch: DenseMatrix[Double] = {

      var count = 0
      var batch: List[DenseVector[Double]] = Nil
      while (iter.hasNext && count < batchSize) {
        count += 1
        val (i, j) = iter.next()
        batch ::= DenseVector(i * delta, j * delta)
      }
      DenseMatrix(batch:_*).t
    }
    override def hasNext: Boolean = iter.hasNext
  }

  override def iterator(batchSize: Int): MeshIterator = new Uniform2DMeshIterator(batchSize)

  override lazy val allPoints: DenseMatrix[Double] = {
    val points = for ( (i,j) <- data) yield DenseVector(i * delta, j * delta)
    DenseMatrix(points:_*).t
  }
}
