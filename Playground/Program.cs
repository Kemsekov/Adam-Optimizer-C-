using System.Diagnostics;
using ScottPlot;
using SysColor = System.Drawing.Color;
var watch = new Stopwatch();
watch.Start();
Examples.NeuralNetworkContinuousLearningExample();
System.Console.WriteLine("Done in " + watch.ElapsedMilliseconds);

