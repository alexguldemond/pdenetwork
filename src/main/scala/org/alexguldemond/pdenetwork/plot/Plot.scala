package org.alexguldemond.pdenetwork.plot

trait Plot[-A] {
  def plot(title: String, a: A): Unit
}

object Plot {

  def plot[A: Plot](title: String, a: A) = implicitly[Plot[A]].plot(title, a)

  implicit class PlotOps[A: Plot](a: A) {
    def plot(title: String) = Plot.plot(title, a)
  }


}