package org.alexguldemond.pdenetwork.model

import breeze.linalg._
import org.alexguldemond.pdenetwork.network.{MultiIndex, NetworkDerivative, WeightVector}

trait SimpleModel extends Model {

  def derivativeMap: Map[MultiIndex, NetworkDerivative]

  def operatorCoefficients(input: DenseMatrix[Double]): MultiIndexCoefficiants

  override def diffOpBatch(input: DenseMatrix[Double]): Transpose[DenseVector[Double]] = {
      val derivatives = derivativeMap
      val coefficients = operatorCoefficients(input)
      diffOpBatch(input, derivatives,coefficients)
  }

  private[this] def diffOpBatch(input: DenseMatrix[Double],
                                derivatives: Map[MultiIndex, NetworkDerivative],
                                coefficients: MultiIndexCoefficiants) = {

    def default: Transpose[DenseVector[Double]] = DenseVector.zeros[Double](input.cols).t

    val result: Transpose[DenseVector[Double]] = derivatives.keys.map{ multiIndex => {
      val co = coefficients.coef.getOrElse(multiIndex, default)
      val vec: Transpose[DenseVector[Double]] = derivatives.get(multiIndex).map( _.applyBatch(input)).getOrElse(default)
      co *:* vec
    }}.reduce(_ + _)

    result :+= coefficients.constant

  }

  override def costGradientBatch(input: DenseMatrix[Double]): WeightVector = {
    val derivatives = derivativeMap
    val coefficients = operatorCoefficients(input)

    val diffOpMinusData = diffOpBatch(input, derivatives, coefficients) - data(input)

    def default: Transpose[DenseVector[Double]] = DenseVector.zeros[Double](input.cols).t

    val weightGradient = derivatives.keys.map{ multiIndex => {
      val coefficient = coefficients.coef.getOrElse(multiIndex, default)
      val weightGradient = derivatives.get(multiIndex).map(_.weightGradientBatch(input)).get
      weightGradient.elemProd(coefficient)
    }}.reduce(_.inPlaceSum(_))

    weightGradient.dotSum(diffOpMinusData)
  }

}
