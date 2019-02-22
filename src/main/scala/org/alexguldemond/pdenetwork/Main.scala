package org.alexguldemond.pdenetwork

import breeze.linalg._
import breeze.numerics.constants.Pi
import breeze.numerics.{exp, sin}
import org.jzy3d.chart._
import org.jzy3d.colors.{Color, ColorMapper}
import org.jzy3d.colors.colormaps.ColorMapRainbow
import org.jzy3d.maths.Range
import org.jzy3d.plot3d.builder.{Builder, Mapper}
import org.jzy3d.plot3d.builder.concrete.OrthonormalGrid
import org.jzy3d.plot3d.rendering.canvas.Quality

object Main extends App {

  val mesh = Uniform2DMesh(.2)
  val model = SimpleLaplacianModel.randomModel(5, 3)

  val epochs = 2
  val batchSize = 10

  time {
    for (i <- 1 to epochs) {
      val iter = mesh.iterator(batchSize)
      model.fit(iter)
      println(s"Epoch $i complete")
    }
  }

  // Define a function to plot
  val exact = new Mapper() {
    def f(x1: Double, x2: Double) =1d/(exp(Pi) - exp(-Pi)) * sin(Pi*x1) * ((exp(Pi*x2) - exp(-Pi * x2)))
  }

  val approx = new Mapper() {
    def f(x1: Double, x2: Double) = model(DenseVector(x1,x2))
  }

  // Define range and precision for the function to plot
  val range = new Range(0, 1)
  val steps = 50

  // Create a surface drawing that function
  val exactSurface =
    Builder.buildOrthonormal(new OrthonormalGrid(range, steps), exact)
  exactSurface.setColorMapper(
    new ColorMapper(new ColorMapRainbow(),
      exactSurface.getBounds().getZmin(),
      exactSurface.getBounds().getZmax(),
      new Color(1, 1, 1, .5f)))
  exactSurface.setFaceDisplayed(true)
  exactSurface.setWireframeDisplayed(false)
  exactSurface.setWireframeColor(Color.BLACK)

  val approxSurface =
    Builder.buildOrthonormal(new OrthonormalGrid(range, steps), approx)
  approxSurface.setColorMapper(
    new ColorMapper(new ColorMapRainbow(),
      approxSurface.getBounds().getZmin(),
      approxSurface.getBounds().getZmax(),
      new Color(1, 1, 1, .5f)))
  approxSurface.setFaceDisplayed(true)
  approxSurface.setWireframeDisplayed(false)
  approxSurface.setWireframeColor(Color.BLACK)

  // Create a chart and add the surface
  val exactPlot = new AWTChart(Quality.Advanced)
  exactPlot.add(exactSurface)
  exactPlot.open("Exact Solution", 600, 600)

  val approxPlot = new AWTChart(Quality.Advanced)
  approxPlot.add(approxSurface)
  approxPlot.open("Approx Solution", 600, 600)


  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) + "ns")
    result
  }

}
