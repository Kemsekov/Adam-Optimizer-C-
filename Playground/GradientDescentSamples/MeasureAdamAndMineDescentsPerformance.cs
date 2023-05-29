using System.Diagnostics;

namespace Playground;
public partial class Examples
{
    public static double MeasureDescentPerformance(Func<IDataAccess<double>, double> problem, int variablesLength, Func<IDataAccess<double>, IGradientDescent> descent, Random? rand = null)
    {
        rand ??= new Random();
        var maxIterations = 20;
        var variables = new ArrayDataAccess<double>(variablesLength);
        for (int k = 0; k < variables.Length; k++)
            variables[k] = rand.NextDouble() * 2 - 1;

        var descentImpl = descent(variables);
        descentImpl.Descent(maxIterations);
        return problem(variables);

    }
    delegate IGradientDescent DescentFactory(Func<IDataAccess<double>, double> problem, IDataAccess<double> parameters);
    public static void MeasureDescentsPerformanceAvg()
    {
        var descents =
        new Dictionary<string,DescentFactory>(){
            {"adam",(problem, parameters)=>new AdamDescent(parameters,problem)},
            {"mine",(problem,parameters)=>new MineDescent(parameters,problem)},
            {"natural",(problem,parameters)=>new NaturalDescent(parameters,problem){ExpectationsSampleCount=20}}
        };
        var descentResults = new Dictionary<string,double[]>();
        var bestDescentCounter = new Dictionary<string,int>();

        var descentTimers = new Dictionary<string,Stopwatch>();

        foreach(var d in descents){
            bestDescentCounter[d.Key] = 0;
            descentTimers[d.Key] = new Stopwatch();
        }

        for (int i = 0; i < 500; i++)
        {
            foreach (var descent in descents)
            {
                var timer = descentTimers[descent.Key];
                var name = descent.Key;
                var factory = descent.Value;

                timer.Start();
                var descentResult = Functions
                    .Select(problem =>
                        MeasureDescentPerformance(problem, 3, parameters => factory(problem,parameters), new Random(i)))
                    .ToArray();
                timer.Stop();
                descentResults[name] = descentResult;
            }
            for(int k = 0;k<Functions.Length;k++){
                var best = descentResults.MinBy(x=>x.Value[k]);
                bestDescentCounter[best.Key]+=1;
            }
        }

        System.Console.WriteLine("Descent\t\tScore\t\tTime ms");
        foreach(var bestDescent in bestDescentCounter){
            var timer = descentTimers[bestDescent.Key];
            System.Console.WriteLine($"{bestDescent.Key}\t\t{bestDescent.Value}\t\t{timer.ElapsedMilliseconds}");
        }
    }
}