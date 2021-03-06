package org.alexguldemond.pdenetwork.plot

import breeze.linalg.DenseVector
import org.alexguldemond.pdenetwork.model.Model
import org.jzy3d.chart.AWTChart
import org.jzy3d.colors.colormaps.ColorMapRainbow
import org.jzy3d.colors.{Color, ColorMapper}
import org.jzy3d.maths.Range
import org.jzy3d.plot3d.builder.concrete.OrthonormalGrid
import org.jzy3d.plot3d.builder.{Builder, Mapper}
import org.jzy3d.plot3d.primitives.Shape
import org.jzy3d.plot3d.rendering.canvas.Quality

object Jzy3dPlotting {

  implicit object ModelPlotImpl extends Plot[Model] {
    def plot(title: String, model: Model) = {

      // Define range and precision for the function to plot
      val range = new Range(0, 1)
      val steps = 50

      val approx = new Mapper() {
        def f(x1: Double, x2: Double) = model(DenseVector(x1,x2))
      }

      val approxSurface: Shape =
        Builder.buildOrthonormal(new OrthonormalGrid(range, steps), approx)
      approxSurface.setColorMapper(
        new ColorMapper(new ColorMapRainbow(),
          approxSurface.getBounds().getZmin(),
          approxSurface.getBounds().getZmax(),
          new Color(1, 1, 1, .5f)))
      approxSurface.setFaceDisplayed(true)
      approxSurface.setWireframeDisplayed(true)
      approxSurface.setWireframeColor(Color.BLACK)

      val approxPlot = new AWTChart(Quality.Advanced)
      approxPlot.add(approxSurface)
      approxPlot.addMouseCameraController()
      approxPlot.open(title, 600, 600)
    }
  }

  implicit object BinaryDoubleFunctionPlotImpl extends Plot[Mapper] {
    def plot(title: String, mapper: Mapper) = {

      // Define range and precision for the function to plot
      val range = new Range(0, 1)
      val steps = 50

      val approxSurface =
        Builder.buildOrthonormal(new OrthonormalGrid(range, steps), mapper)
      approxSurface.setColorMapper(
        new ColorMapper(new ColorMapRainbow(),
          approxSurface.getBounds().getZmin(),
          approxSurface.getBounds().getZmax(),
          new Color(1, 1, 1, .5f)))
      approxSurface.setFaceDisplayed(true)
      approxSurface.setWireframeDisplayed(false)
      approxSurface.setWireframeColor(Color.BLACK)


      val approxPlot = new AWTChart(Quality.Advanced)
      approxPlot.add(approxSurface)
      approxPlot.addMouseCameraController()
      approxPlot.open(title, 600, 600)
    }
  }
}
