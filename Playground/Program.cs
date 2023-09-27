using System.Diagnostics;
using System.Numerics;
using System.Xml;
using GradientDescentSharp.Utils.Kernels;
using MathNet.Numerics.LinearAlgebra.Double;

var watch = new Stopwatch();
watch.Start();
// Examples.NeuralNetworkClassificationExample();
// Examples.CompareAdamAndMineDescent(Examples.Functions[1],3);
var A = DenseMatrix.Create(10,10,(i,j)=>Random.Shared.NextDouble()*4-2);

var b = DenseVector.Create(10,i=>Random.Shared.NextSingle()*2);

var x = new ArrayDataAccess<double>(10);

var solution = A.Solve(b);

for(int i = 0;i<x.Length;i++)
    x[i] = b[i];
System.Console.WriteLine(loss(x));
double loss(IDataAccess<double> x){
    var xVec = new ComplexObjectsFactory(x).CreateVector(10);
    var res =A*xVec;
    return (b-res).L2Norm();
}

var descent = new MineDescent(x,loss){
    Logger = new ConsoleLogger(),
    DescentRate=10,
    Theta=0,
    DescentRateDecreaseRate=0.5
};
descent.Descent(40);
System.Console.WriteLine(loss(x));

System.Console.WriteLine(new ComplexObjectsFactory(x).CreateVector(10));
System.Console.WriteLine(solution);

System.Console.WriteLine("Done in " + watch.ElapsedMilliseconds);
