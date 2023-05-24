using System.Diagnostics;
using System.Globalization;
using CsvHelper;
using GradientDescentSharp.NeuralNetwork;
using Playground.DataModels;

var watch = new Stopwatch();
watch.Start();
Examples.NeuralNetworkClassificationExample();
System.Console.WriteLine("Done in " + watch.ElapsedMilliseconds);

