package org.alexguldemond.pdenetwork

import breeze.linalg._

import scala.collection.mutable.ListBuffer

trait Model {

  def cost(input: DenseVector[Double]): Double = batchCost(input.asDenseMatrix.t)

  def batchCost(input: DenseMatrix[Double]): Double = {
    val l = diffOpBatch(input) - data(input)
    sum((l *:* l) / 2d)
  }

  def averageCost(input: DenseMatrix[Double]): Double = batchCost(input) / (input.cols).toDouble

  def diffOpBatch(input: DenseMatrix[Double]): Transpose[DenseVector[Double]]

  def diffOp(input: DenseVector[Double]): Double = diffOpBatch(input.asDenseMatrix.t)(0)

  def data(input: DenseMatrix[Double]): Transpose[DenseVector[Double]]

  def costGradient(input: DenseVector[Double]): WeightGradient = costGradientBatch(input.asDenseMatrix.t)

  def costGradientBatch(input: DenseMatrix[Double]): WeightGradient

  def averageGradient(input: DenseMatrix[Double]): WeightGradient = costGradientBatch(input) / input.cols.toDouble

  def apply(input: DenseVector[Double]): Double

  def update(input: DenseMatrix[Double], updater: Updater): Unit = updater.updateModel(this, input)

  def updateWeights(weightGradient: WeightGradient): Unit

  def fit(iter: MeshIterator, updater: Updater): Unit = {
    while (iter.hasNext) {
      val batch = iter.nextBatch
      update(batch, updater)
    }
  }

  def fit(iter: MeshIterator, reportFrequency: Int, updater: Updater): ListBuffer[Double] = {
    var counter = 1
    var report: ListBuffer[Double] = new ListBuffer[Double]()
    while (iter.hasNext) {
      val batch = iter.nextBatch
      if (counter % reportFrequency == 0) {
        report += averageCost(batch)
      }
      update(batch, updater)
      counter = counter + 1
    }
    report
  }
}

case class MultiIndexCoefficiants(constant: Transpose[DenseVector[Double]],
                                  coef: Map[MultiIndex,Transpose[DenseVector[Double]] ])
