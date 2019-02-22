package org.alexguldemond.pdenetwork

import breeze.linalg.{DenseMatrix, DenseVector}

trait Model {

  def cost(input: DenseVector[Double]): Double

  def batchCost(input: DenseMatrix[Double]): Double

  def averageCost(input: DenseMatrix[Double]): Double = batchCost(input) / (input.cols).toDouble

  def costGradient(input: DenseVector[Double]): WeightGradient

  def costGradientBatch(input: DenseMatrix[Double]): WeightGradient

  def apply(input: DenseVector[Double]): Double

  def update(input: DenseMatrix[Double]): Unit

  def fit(iterator: MeshIterator): Unit
}
