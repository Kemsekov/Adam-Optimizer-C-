namespace GradientDescentSharp.GradientDescents;

public abstract class GradientDescentBase : IGradientDescent
{
    /// <summary>
    /// Error function that we need to minimize. Returns values >= 0
    /// </summary>
    /// <value></value>
    public Func<IDataAccess<double>, double> Function { get; }
    /// <summary>
    /// Function variables that need to be adjusted to minimize error function 
    /// </summary>
    /// <value></value>
    public IDataAccess<double> Variables { get; }
    /// <summary>
    /// Length of function variables
    /// </summary>
    public int Dimensions { get; }
    /// <summary>
    /// Descent rate. The higher, the more.
    /// </summary>
    public double DescentRate = 0.05;
    /// <summary>
    /// If on gradient descent step error function value chang less than this value, it means
    /// we need to step descending. The lower this value, the more precise descending will hit local minima.
    /// </summary>
    public double Theta = 0.0001;
    /// <summary>
    /// Create new instance of gradient descent
    /// </summary>
    /// <param name="variables">Variables that will be adjusted</param>
    /// <param name="function">
    /// Error function.<br/>
    /// Good rule of thumb: return squared error for each variable, so
    /// if you have for example 2 variables, then return it like this<br/>
    /// error = (x[0]-x0Error)^2+(x[1]-x1Error)^2<br/>
    /// </param>
    public GradientDescentBase(IDataAccess<double> variables, Func<IDataAccess<double>, double> function)
    {
        Function = function;
        Variables = variables;
        Dimensions = variables.Length;
    }
    protected double Evaluate(IDataAccess<double> variables)
    {
        return Function(variables);
    }
    protected void Step(IDataAccess<double> change)
    {
        for (var i = 0; i < Dimensions; i++)
            Variables[i] -= change[i];
    }
    protected void UndoStep(IDataAccess<double> change)
    {
        for (var i = 0; i < Dimensions; i++)
            Variables[i] += change[i];
    }
    protected void ComputeGradient(IDataAccess<double> gradient, double currentEvaluation)
    {
        Parallel.For(0, Dimensions, i =>
        {
            var gradientDataAccess = new GradientDataAccess<double>(Variables, 0, 0);
            gradientDataAccess.Reset(i, Variables[i] + Theta);
            var after = Evaluate(gradientDataAccess);
            gradient[i] = (after - currentEvaluation) / Theta;
        });
    }
    public abstract int Descent(int maxIterations);
}