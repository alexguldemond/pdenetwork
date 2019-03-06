package org.alexguldemond.pdenetwork.activation

import breeze.generic.{MappingUFunc, UFunc}

object ArcTanDerivatives {
  object tanhFirstDerivative extends UFunc with MappingUFunc {
    implicit object tanhImplDouble extends Impl[Double, Double] {
      def apply(x:Double): Double = {
        1d/(1d + x*x)
      }
    }
  }

  object tanhSecondDerivative extends UFunc with MappingUFunc {
    implicit object tanhImplDouble extends Impl[Double, Double] {
      def apply(x:Double): Double = {
        val onePlusXSquared = 1d + x*x
        -2d * x / (onePlusXSquared * onePlusXSquared)
      }
    }
  }

  object tanhThirdDerivative extends UFunc with MappingUFunc {
    implicit object tanhImplDouble extends Impl[Double, Double] {
      def apply(x:Double): Double = {
        val onePlusXSquared = 1d + x*x
        val onePlusXSquared2 = onePlusXSquared * onePlusXSquared
        8d * x * x / (onePlusXSquared2 * onePlusXSquared) - 2d / onePlusXSquared2
      }
    }
  }

  object tanhFourthDerivative extends UFunc with MappingUFunc {
    implicit object tanhImplDouble extends Impl[Double, Double] {
      def apply(x:Double): Double = {
        val onePlusXSquared = 1d + x*x
        val onePlusXSquared3 = onePlusXSquared * onePlusXSquared * onePlusXSquared
        -48d * x*x*x / (onePlusXSquared3 * onePlusXSquared) + 24d * x/onePlusXSquared3
      }
    }
  }

}
