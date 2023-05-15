# GradientDescentSharp
This little library allows to compute a approximate solution for some defined problem with error function, using gradient descent

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
Error is 3.7554352527960218
Changed by 0.13403050614940248
-------------
Error is 3.5935433267621137
Changed by 0.161891926033908
-------------
Error is 3.398307101797787
Changed by 0.1952362249643267
-------------
Error is 3.162882692912615
Changed by 0.23542440888517202
-------------
Error is 2.878749428902407
Changed by 0.2841332640102081
-------------
Error is 2.535305100515128
Changed by 0.343444328387279
-------------
Error is 2.1193499029316687
Changed by 0.4159551975834592
-------------
Error is 1.6144296560815143
Changed by 0.5049202468501544
-------------
Error is 0.9999999999999938
Changed by 0.6144296560815206
-------------
Error is 0.2503619082577506
Changed by 0.7496380917422432
-------------
Error is 0.66669577875009
Changed by 0.4163338704923394
Undo step. Decreasing descentRate.
-------------
Error is 0.16678649104883014
Changed by 0.08357541720892048
-------------
Error is 0.0815073677144893
Changed by 0.08527912333434085
-------------
Error is 0.005512558460064376
Changed by 0.07599480925442492
-------------
Error is 0.0815073677144893
Changed by 0.07599480925442492
Undo step. Decreasing descentRate.
-------------
Error is 0.003268845143455934
Changed by 0.002243713316608442
-------------
Error is 0.005512558460064376
Changed by 0.002243713316608442
Undo step. Decreasing descentRate.
-------------
Error is 0.002391504097262853
Changed by 0.0008773410461930808
-------------
Error is 0.0015139855279064918
Changed by 0.0008775185693563614
-------------
Error is 0.0006362893970743855
Changed by 0.0008776961308321063
-------------
Error is 0.00024158433355214726
Changed by 0.0003947050635222382
-------------
Error is 0.0006362893970743855
Changed by 0.0003947050635222382
Undo step. Decreasing descentRate.
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
