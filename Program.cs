using AdamOptimizer;
Func<double, double, double, double>[] functions ={
     (a,b,c)=>a-b*c+Math.Exp(a*b*c)-Math.Sin(a+b+c),
     (a,b,c)=>a-b*c,
     (a,b,c)=>a*b*c-a*Math.Sin(c)+Math.Cos(a*b)+c,
     (a,b,c)=>a*Math.Cosh(a*b+Math.Sin(c)),
     (a,b,c)=>Math.Tan(a*b+c)-Math.Sin(b*c+a),
     (a,b,c)=>Math.Pow(a*a,b)-c*Math.Exp(a)*Math.Log(c*c)
};

//choose function to find it's minima
// var func = (double a, double b, double c)=>Math.Abs(functions.Sum(x=>x(a,b,c)));
var func = (double a, double b, double c) => Math.Abs(functions[5](a, b, c));

var functionToFeed = (IDataAccess<double> x) => func(x[0], x[1], x[2]);

ArrayDataAccess<double> variables1 = new double[] { Random.Shared.NextDouble(), Random.Shared.NextDouble(), Random.Shared.NextDouble() };
ArrayDataAccess<double> variables2 = new double[3];
variables1.Array.CopyTo(variables2, 0);

var gradientDescent1 = new GradientDescent(variables1, functionToFeed);
var gradientDescent2 = new GradientDescent(variables2, functionToFeed);

var before = func(variables1[0], variables1[1], variables1[2]);

var maxIterations = 40;
var learningRate = 0.1;
var theta = 0.0001;

System.Console.WriteLine(gradientDescent1.AdamDescent(maxIterations, learningRate, theta) + " Adam iterations");
System.Console.WriteLine(gradientDescent2.MineDescent(maxIterations, learningRate, theta) + " Mine iterations");
var after1 = functionToFeed(variables1);
var after2 = functionToFeed(variables2);

Console.WriteLine("Before " + before);
Console.WriteLine("Adam " + after1);
Console.WriteLine("Mine " + after2);
System.Console.WriteLine("Adam Point is [" + String.Join(' ', variables1) + "]");
System.Console.WriteLine("Mine Point is [" + String.Join(' ', variables2) + "]");

MeasurePerformance.Measure(functionToFeed);

