using System.Diagnostics;
using GradientDescentSharp.NeuralNetwork;
using ScottPlot;
var watch = new Stopwatch();
watch.Start();
Examples.NeuralNetworkOnDatasetExample();
System.Console.WriteLine("Done in "+watch.ElapsedMilliseconds);