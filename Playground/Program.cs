using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra.Double;
using ILGPU;

var watch = new Stopwatch();
watch.Start();
Examples.LinearRegression();



var Context = ILGPU.Context.Create(builder =>
{
    builder.Optimize(OptimizationLevel.O2);
    builder.Default();
});
// Accelerator = Context.GetPreferredDevice(true).CreateAccelerator(Context);
var d = Context.GetPreferredDevices(false, false);
System.Console.WriteLine("Done in " + watch.ElapsedMilliseconds);


