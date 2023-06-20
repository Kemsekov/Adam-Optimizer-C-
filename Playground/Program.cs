using System.Diagnostics;
using GradientDescentSharp.Utils.Kernels;
using MathNet.Numerics.LinearAlgebra.Double;

var watch = new Stopwatch();
watch.Start();
Examples.NeuralNetworkClassificationExample();
System.Console.WriteLine("Done in " + watch.ElapsedMilliseconds);

