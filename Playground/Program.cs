using System.Diagnostics;

var watch = new Stopwatch();
watch.Start();
Examples.MeasureDescentsPerformanceAvg();
System.Console.WriteLine("Done in " + watch.ElapsedMilliseconds);

