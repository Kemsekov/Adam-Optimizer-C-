using System.Diagnostics;
var watch = new Stopwatch();
watch.Start();
Examples.NeuralNetworkExample();
System.Console.WriteLine("Done in " + watch.ElapsedMilliseconds);

