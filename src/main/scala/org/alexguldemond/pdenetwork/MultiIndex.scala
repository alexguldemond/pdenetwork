package org.alexguldemond.pdenetwork

import breeze.linalg.DenseVector

case class MultiIndex(values: Seq[Int]) {

  def dim = values.size

  lazy val total: Int = values.reduce(_ + _)

  def apply(index: Int) = values(index)

  def asVector = DenseVector(values.map( _.toDouble).toArray)
}
