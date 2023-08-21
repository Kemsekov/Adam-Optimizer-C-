namespace GradientDescentSharp.GradientDescents;
/// <summary>
/// Base class for gradient descent implementations
/// </summary>
public abstract class GradientDescentBase : IGradientDescent
{
    /// <summary>
    /// Parameters: <br/> First is gradient output.<br/> 
    /// You need to set each value under each index to a gradient of a <see cref="Function"/> <br/> <br/> 
    /// Second is variables <br/> 
    /// So you can tweak each of them, reevaluate function, compute gradients - or just use derivative formula.<br/> <br/> 
    /// Third is current evaluation of function <see cref="Function"/> at the second parameter(variables) <br/> <br/> 
    /// By default this method is set to find gradient numerically - which is slow <br/> 
    /// So if you have a analytic gradient formula, then redefine this action and get fast performance.
    /// </summary>
    public Action<IDataAccess<double>, IDataAccess<double>, double> ComputeGradient{get;set;}
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
    /// If gradient descent step error function value changes less than this value, it means
    /// we need to step descending. <br/>
    /// In short terms, this value defines how precise our find local minima need to be.<br/>
    /// Set to zero, if you need to find the most precise possible solution
    /// </summary>
    public double Theta = 0.0001;
    /// <summary>
    /// This value is a derivative epsilon, that used to compute gradient
    /// </summary>
    public double Epsilon = 0.0001;
    /// <summary>
    /// Create new instance of gradient descent
    /// </summary>
    /// <param name="variables">Variables that will be adjusted</param>
    /// <param name="function">
    /// Error function.<br/>
    /// Good rule of thumb: return MSE as error function<br/>
    /// So if your error values is a,b,c,d => then return error as: a^2+b^2+c^2+d^2
    /// </param>
    public GradientDescentBase(IDataAccess<double> variables, Func<IDataAccess<double>, double> function)
    {
        Function = function;
        Variables = variables;
        Dimensions = variables.Length;
        ComputeGradient = ComputeGradientDefaultImplementation;
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
    /// <summary>
    /// Computes a gradient of error function by nudging it by theta.
    /// </summary>
    protected void ComputeGradientDefaultImplementation(IDataAccess<double> gradient, IDataAccess<double> variables, double currentEvaluation)
    {
        Parallel.For(0, Dimensions, i =>
        {
            var gradientDataAccess = new GradientDataAccess<double>(variables, i, variables[i] + Epsilon);
            var after = Evaluate(gradientDataAccess);
            gradient[i] = (after - currentEvaluation) / Epsilon;
        });
    }
    public abstract int Descent(int maxIterations);
}