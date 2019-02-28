package org.alexguldemond.pdenetwork

import breeze.plot.Figure
import breeze.plot.{plot => bPlot}

object BreezePlotting {


  implicit object ListPlotImpl extends Plot[List[Double]] {
    def plot(title: String, list: List[Double]) = {
      import java.awt.Graphics2D
      import java.awt.image.BufferedImage
      val image = new BufferedImage(600, 600, BufferedImage.TYPE_INT_ARGB)
      val graphics2D : Graphics2D= image.createGraphics

      val figure = Figure(title)
      val p = figure.subplot(0)
      p += bPlot(list.zipWithIndex.map(_._2.toDouble), list)
      p.xlabel = "Iteration"
      p.ylabel = "Cost"
      figure.drawPlots(graphics2D)
    }
  }
}
