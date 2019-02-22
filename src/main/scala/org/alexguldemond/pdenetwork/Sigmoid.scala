package org.alexguldemond.pdenetwork

import breeze.generic.{MappingUFunc, UFunc}
import breeze.numerics.{exp, pow, sigmoid}

object SigmoidDerivatives {

  object sigmoidFirstDerivative extends UFunc with MappingUFunc {
    implicit object sigmoidImplDouble extends Impl[Double, Double] {
      def apply(x:Double): Double = {
        val sigma = sigmoid(x)
        sigma*(1 - sigma)
      }
    }
  }

  object sigmoidSecondDerivative extends UFunc with MappingUFunc {
    implicit object sigmoidImplDouble extends Impl[Double, Double] {
      def apply(x:Double): Double = {
        val sigma = sigmoid(x)
        sigma*(1 - sigma)*(1-2*sigma)
      }
    }
  }

  object sigmoidThirdDerivative extends UFunc with MappingUFunc {
    implicit object sigmoidImplDouble extends Impl[Double, Double] {
      def apply(x:Double): Double = {
        val sigma = sigmoid(x)
        sigma*(1 - sigma)*(1 - 6*sigma + 6*sigma*sigma)
      }
    }
  }

  object sigmoidFourthDerivative extends UFunc with MappingUFunc {
    implicit object sigmoidImplDouble extends Impl[Double, Double] {
      def apply(x:Double): Double = {
        val ex = exp(x)
        -ex * (-1 + 11*ex -11*ex*ex + ex*ex*ex)/(pow(1 + ex, 5))
      }
    }
  }

}
