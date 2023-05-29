[![nuget](https://img.shields.io/nuget/v/GradientDescentSharp.svg)](https://www.nuget.org/packages/GradientDescentSharp/) 
# GradientDescentSharp
This little library allows to compute a approximate solution for some defined problem with error function, using gradient descent.

Simple example:
```cs
//first define a problem
var problem = (IDataAccess<double> x) =>
{
    var n = x[0];
    //we seek for such value, that n^n=5
    var needToMinimize = Math.Pow(n, n) - 5.0;
    return Math.Abs(needToMinimize);
};
//then define changing variables
var variables = new ArrayDataAccess<double>(1);
//set variables close to global minima
variables[0] = 1;

//define descent
var descent = new MineDescent(variables, problem)
{
    DescentRate = 0.1,             // how fast to descent, this value will be adjusted on the fly
    Theta = 1e-4,                  // what precision of found minima we need
    DescentRateDecreaseRate = 0.1, // how much decrease DescentRate when we hit a grow of error function
    Logger = new ConsoleLogger()   // logger for descent progress
};

//do 30 iterations
descent.Descent(30);

System.Console.WriteLine("For problem n^n=5");
System.Console.WriteLine($"Error is {problem(variables)}");
System.Console.WriteLine($"n={variables[0]}");
System.Console.WriteLine($"n^n={Math.Pow(variables[0], variables[0])}");
```

Output 
```
--------------Mine descent began
Error is 3.8894657589454242
Changed by 0.11053424105457577
-------------
...
-------------
Error is 0.2503619082577506
Changed by 0.7496380917422432
-------------
Error is 0.66669577875009
Changed by 0.4163338704923394
Undo step. Decreasing descentRate.
-------------
...
-------------
Error is 0.00015378896740614323
Changed by 8.779536614600403E-05
--------------Mine done in 24 iterations
For problem n^n=5
Error is 0.00015378896740614323
n=2.1293900000000012
n^n=5.000153788967406
```

As you can see, when we hit a grow in error function, the descent undo step and decrease descentRate, so we are guaranteed to hit a local minima!

It still is very important to define a good error function and init variables.

Currently there are three providers for gradient descent:
adam descent
mine descent
natural descent

Here is general time-performance measurements for each of them
```text
Descent         Score           Time ms
adam            290             270
mine            971             162
natural         1739            293
```

As you can see mine descent currently is fastest one and provides a good results, while natural descent is the most precise

Also I've a good working feed-forward neural network implementation here.

Check out 
[gradient descent playground examples](https://github.com/Kemsekov/GradientDescentSharp/tree/main/Playground/GradientDescentSamples).

Check out 
[neural network playground examples](https://github.com/Kemsekov/GradientDescentSharp/tree/main/Playground/NeuralNetworkSamples).
