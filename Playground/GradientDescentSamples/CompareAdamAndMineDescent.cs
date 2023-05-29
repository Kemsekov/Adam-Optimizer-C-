namespace Playground;
public partial class Examples
{
    public static void CompareAdamAndMineDescent(Func<IDataAccess<double>, double> func, int variablesLength, Action<IDataAccess<double>>? init = null)
    {
        //choose function to find it's minima
        // var func = (double a, double b, double c)=>Math.Abs(Functions.Sum(x=>x(a,b,c)));

        ArrayDataAccess<double> variables1 = new double[variablesLength];
        ArrayDataAccess<double> variables2 = new double[variablesLength];

        if (init is null)
            for (int i = 0; i < variablesLength; i++)
                variables1[i] = Random.Shared.NextDouble() * 2 - 1;
        init?.Invoke(variables1);

        variables1.Array.CopyTo(variables2, 0);
        var maxIterations = 20;
        var descentRate = 0.1;
        var theta = 0.0001;
        var logger = new ConsoleLogger();
        var adamDescent = new AdamDescent(variables1, func)
        {
            Theta = theta,
            DescentRate = descentRate,
            Logger = logger
        };
        var mineDescent = new MineDescent(variables2, func)
        {
            Theta = theta,
            DescentRate = descentRate,
            Logger = logger
        };

        var before = func(variables1);
        adamDescent.Descent(maxIterations);
        mineDescent.Descent(maxIterations);

        var after1 = func(variables1);
        var after2 = func(variables2);

        Console.WriteLine("Before " + before);
        Console.WriteLine("Adam " + after1);
        Console.WriteLine("Mine " + after2);
        System.Console.WriteLine("Adam Point is [" + String.Join(' ', variables1.Select(x => x.ToString("0.000"))) + "]");
        System.Console.WriteLine("Mine Point is [" + String.Join(' ', variables2.Select(x => x.ToString("0.000"))) + "]");
    }
}