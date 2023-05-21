using System.Diagnostics;
using GradientDescentSharp.NeuralNetwork;
using MathNet.Numerics.LinearAlgebra.Single;

var watch = new Stopwatch();
watch.Start();
Examples.NeuralNetworkExample();
System.Console.WriteLine("Done in " + watch.ElapsedMilliseconds);