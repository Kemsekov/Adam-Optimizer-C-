namespace GradientDescentSharp;
/// <summary>
/// Simply takes a problem and gradient descent implementation, generate a lot
/// of solutions and returns the best one.
/// </summary>
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
    public Func<IDataAccess<FloatType>, IGradientDescent> GradientDescentFactory { get; }
    /// <summary>
    /// How many solutions to build
    /// </summary>
    public int SolutionsCount = 100;
    /// <summary>
    /// Input parameters initialization function
    /// </summary>
    public Action<IDataAccess<FloatType>>? Init = null;
    /// <summary>
    /// How many descent iterations need to do on each solution
    /// </summary>
    public int DescentIterations = 100;
    /// <param name="variablesLength">How long input data is</param>
    /// <param name="gradientDescentFactory">Function that takes input data and returns gradient descent implementation</param>
    /// <param name="init">Weight initialization. Use it to generate a starting position that is closer to global minima and so can converge better</param>
    /// <param name="descentIterations">How many descent steps each solution need to take</param>
    /// <param name="solutionsCount">How many solutions need to crete</param>
    public BestSolutionFinder(int variablesLength, Func<IDataAccess<FloatType>, IGradientDescent> gradientDescentFactory, Action<IDataAccess<FloatType>>? init = null,int descentIterations = 100,int solutionsCount = 100)
    {
        VariablesLength = variablesLength;
        GradientDescentFactory = gradientDescentFactory;
        Init = init;
        DescentIterations = descentIterations;
        SolutionsCount = solutionsCount;
    }
    /// <summary>
    /// Generates a lot of solutions, init each of them, finds local minima and returns the best one of all.
    /// </summary>
    public IDataAccess<FloatType> TryToFindBestSolution(Func<IDataAccess<FloatType>, FloatType> func)
    {
        Init ??= variables =>
            {
                for (int i = 0; i < VariablesLength; i++)
                    variables[i] = Random.Shared.NextDouble() * 4 - 2;
            };
        
        IDataAccess<FloatType> variables = new ArrayDataAccess<FloatType>(VariablesLength);
        Init?.Invoke(variables);

        ArrayDataAccess<FloatType> bestMine = variables.ToArray();
        
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