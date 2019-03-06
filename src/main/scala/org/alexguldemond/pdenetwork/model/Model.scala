package org.alexguldemond.pdenetwork.model

import breeze.linalg._
import org.alexguldemond.pdenetwork.mesh.Mesh
import org.alexguldemond.pdenetwork.network.{MultiIndex, WeightVector}
import org.alexguldemond.pdenetwork.updater.{FullUpdater, StochasticUpdater, Updater}

import scala.collection.mutable.ListBuffer

trait Model {

  def copyArchitecture(weightVector : DenseVector[Double]): Model

  def weightVector: WeightVector

  def cost(input: DenseVector[Double]): Double = batchCost(input.asDenseMatrix.t)

  def batchCost(input: DenseMatrix[Double]): Double = {
    val l = diffOpBatch(input) - data(input)
    sum((l *:* l) / 2d)
  }

  def averageCost(input: DenseMatrix[Double]): Double = batchCost(input) / (input.cols).toDouble

  def diffOpBatch(input: DenseMatrix[Double]): Transpose[DenseVector[Double]]

  def diffOp(input: DenseVector[Double]): Double = diffOpBatch(input.asDenseMatrix.t)(0)

  def data(input: DenseMatrix[Double]): Transpose[DenseVector[Double]]

  def costGradient(input: DenseVector[Double]): WeightVector = costGradientBatch(input.asDenseMatrix.t)

  def costGradientBatch(input: DenseMatrix[Double]): WeightVector

  def averageGradient(input: DenseMatrix[Double]): WeightVector = costGradientBatch(input) / input.cols.toDouble

  def apply(input: DenseVector[Double]): Double

  def update(input: DenseMatrix[Double], updater: Updater): Model = updater.updateModel(this, input)

  def updateWeights(weightGradient: WeightVector): Unit

  def updateWeights(weightVector: DenseVector[Double]): Unit

  def fit(mesh: Mesh, batchSize: Int, reportFrequency: Int, updater: StochasticUpdater): ListBuffer[Double] = {
    var counter = 0
    var report: ListBuffer[Double] = new ListBuffer[Double]()
    val iter = mesh.iterator(batchSize)
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

  def fit(mesh: Mesh, updater: FullUpdater): Double = {
    val dataPoints = mesh.allPoints
    update(dataPoints, updater)
    averageCost(dataPoints)
  }
}

case class MultiIndexCoefficiants(constant: Transpose[DenseVector[Double]],
                                  coef: Map[MultiIndex,Transpose[DenseVector[Double]] ])
