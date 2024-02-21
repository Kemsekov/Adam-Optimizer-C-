using System.Diagnostics;
using System.Numerics;
using System.Xml;
using GradientDescentSharp.Utils.Kernels;
using MathNet.Numerics.LinearAlgebra.Double;

var watch = new Stopwatch();
watch.Start();
// Examples.NeuralNetworkClassificationExample();
// Examples.CompareAdamAndMineDescent(Examples.Functions[1],3);

var x = new ArrayDataAccess<double>(2);
x[0] = 1;
x[1] = 1;
double loss(IDataAccess<double> x){
    var xReal = x[0];
    var xImaginary = x[1];
    var complex = new Complex(xReal,xImaginary);
    var complexPow2 = complex*complex;
    var complexPow4 = complexPow2*complexPow2;
    return (complexPow4-complexPow2*complex-2*complexPow2+3*complex-3).Magnitude;
}

var descent = new AdamDescent(x,loss){
    Logger = new ConsoleLogger(),
    DescentRate=1,
    Theta=0,
    DescentRateDecreaseRate=0.1
};
double previousLoss = 1;
foreach(var d in descent.Descent().Take(400)){
    var loss1 = d.Loss;
    var lossAvg = (previousLoss+loss1)/2;
    var diff = Math.Abs(1-loss1/lossAvg);
    if(diff<0.1){
        descent.DescentRate*=1.2;
    }
    previousLoss = loss1;
}
System.Console.WriteLine(loss(x));

System.Console.WriteLine(x);

System.Console.WriteLine("Done in " + watch.ElapsedMilliseconds);
