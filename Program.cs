using AdamOptimizer;

var func1 = (double a, double b,double c)=>Math.Abs(a-b*c+Math.Exp(a*b*c)-Math.Sin(a+b+c));
var func2 = (double a, double b,double c)=>Math.Abs(a-b*c);
var func3 = (double a, double b,double c)=>Math.Abs(a*b*c-a*Math.Sin(c)+Math.Cos(a*b)+c);
var func4 = (double a, double b,double c)=>Math.Abs(a*Math.Cosh(a*b+Math.Sin(c)));
var func5 = (double a, double b,double c)=>Math.Abs(Math.Tan(a*b+c)-Math.Sin(b*c+a));
var func6 = (double a, double b,double c)=>Math.Abs(Math.Pow(a*a,b)-c*Math.Exp(a)*Math.Log(c*c));

//choose function to find it's minima
var func = func4;

var functionToFeed = (double[] x)=>func(x[0],x[1],x[2]);

var variables1 = new double[]{Random.Shared.NextDouble(),Random.Shared.NextDouble(),Random.Shared.NextDouble()};
var variables2 =new double[3];
variables1.CopyTo(variables2,0);

var gradientDescend1 = new GradientDescend(variables1,functionToFeed);
var gradientDescend2 = new GradientDescend(variables2,functionToFeed);

var before = func(variables1[0],variables1[1],variables1[2]);

var maxIterations = 40;
var learningRate = 0.1;
var theta = 0.0001;

System.Console.WriteLine(gradientDescend1.AdamDescent(maxIterations,learningRate,theta)+" Adam iterations");
System.Console.WriteLine(gradientDescend2.MineDescent(maxIterations,learningRate,theta)+" Mine iterations");
var after1 = functionToFeed(variables1);
var after2 = functionToFeed(variables2);

Console.WriteLine("Before "+before);
Console.WriteLine("Adam "+after1);
Console.WriteLine("Mine "+after2);
System.Console.WriteLine("Adam Point is ["+String.Join(' ',variables1)+"]");
System.Console.WriteLine("Mine Point is ["+String.Join(' ',variables2)+"]");

MeasurePerformance.Measure(functionToFeed);

