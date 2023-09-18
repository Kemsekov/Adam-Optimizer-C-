using System.Diagnostics;
using System.Numerics;
using GradientDescentSharp.Utils.Kernels;
using MathNet.Numerics.LinearAlgebra.Double;

var watch = new Stopwatch();
watch.Start();
// Examples.NeuralNetworkClassificationExample();
Examples.CompareAdamAndMineDescent(Examples.Functions[1],3);
System.Console.WriteLine("Done in " + watch.ElapsedMilliseconds);
