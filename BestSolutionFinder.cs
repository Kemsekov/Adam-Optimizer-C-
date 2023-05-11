namespace GradientDescentSharp;
public class BestSolutionFinder
{
    /// <summary>
    /// In this logger will be passed information about solutions
    /// </summary>
    public ILogger? Logger;
    /// <summary>
    /// Variables length of error function input
    /// </summary>
    int VariablesLength;
    public Func<IDataAccess<double>, IGradientDescent> GradientDescentFactory { get; }
    /// <summary>
    /// How many solutions to build
    /// </summary>
    public int SolutionsCount = 100;
    /// <summary>
    /// Input parameters initialization function
    /// </summary>
    public Action<IDataAccess<double>>? Init = null;
    /// <summary>
    /// How many descent iterations need to do on each solution
    /// </summary>
    public int DescentIterations = 100;
    public BestSolutionFinder(int variablesLength, Func<IDataAccess<double>, IGradientDescent> gradientDescentFactory, Action<IDataAccess<double>>? init = null)
    {
        VariablesLength = variablesLength;
        GradientDescentFactory = gradientDescentFactory;
        Init = init;
    }

    public IDataAccess<double> TryToFindBestSolution(Func<IDataAccess<double>, double> func)
    {
        Init ??= variables =>
            {
                for (int i = 0; i < VariablesLength; i++)
                    variables[i] = Random.Shared.NextDouble() * 4 - 2;
            };
        
        IDataAccess<double> variables = new ArrayDataAccess<double>(VariablesLength);
        Init?.Invoke(variables);

        ArrayDataAccess<double> bestMine = variables.ToArray();

        var before = func(variables);
        for (int k = 0; k < SolutionsCount; k++)
        {
            Logger?.LogLine("Before " + func(variables));
            var gradient = GradientDescentFactory(variables);
            gradient.Descent(DescentIterations);
            var currentScore = func(variables);
            var bestScore = func(bestMine);
            Logger?.LogLine("After " + currentScore);
            Logger?.LogLine("-----------");
            if (currentScore < bestScore)
            {
                bestMine = variables.ToArray();
            }
            Init?.Invoke(variables);
        }
        var mineScore = func(bestMine);
        Logger?.LogLine("Error is " + mineScore);
        Logger?.LogLine("Variables is [" + string.Join(' ', bestMine.Select(x => x.ToString("0.000"))) + "]");
        return bestMine;
    }
}