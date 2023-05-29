using System.Diagnostics;
using System.Globalization;
using CsvHelper;
using GradientDescentSharp.NeuralNetwork;
using Playground.DataModels;

var watch = new Stopwatch();
watch.Start();
Examples.CompareAdamAndMineDescent(Examples.Functions[0],3);
System.Console.WriteLine("Done in " + watch.ElapsedMilliseconds);

