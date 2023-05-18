namespace GradientDescentSharp.GradientDescents;
/// <summary>
/// Base class for gradient descent implementations
/// </summary>
public abstract class GradientDescentBase : IGradientDescent
{
    /// <summary>
    /// Error function that we need to minimize. Returns values >= 0
    /// </summary>
    /// <value></value>
    public Func<IDataAccess<FloatType>, FloatType> Function { get; }
    /// <summary>
    /// Function variables that need to be adjusted to minimize error function 
    /// </summary>
    /// <value></value>
    public IDataAccess<FloatType> Variables { get; }
    /// <summary>
    /// Length of function variables
    /// </summary>
    public int Dimensions { get; }
    /// <summary>
    /// Descent rate. The higher, the more.
    /// </summary>
    public FloatType DescentRate = 0.05;
    /// <summary>
    /// If gradient descent step error function value changes less than this value, it means
    /// we need to step descending. <br/>
    /// In short terms, this value defines how precise our find local minima need to be.<br/>
    /// Also this value is a derivative epsilon, that used to compute gradient
    /// </summary>
    public FloatType Theta = 0.0001;
    /// <summary>
    /// Create new instance of gradient descent
    /// </summary>
    /// <param name="variables">Variables that will be adjusted</param>
    /// <param name="function">
    /// Error function.<br/>
    /// Good rule of thumb: return MSE as error function<br/>
    /// So if your error values is a,b,c,d => then return error as: a^2+b^2+c^2+d^2
    /// </param>
    public GradientDescentBase(IDataAccess<FloatType> variables, Func<IDataAccess<FloatType>, FloatType> function)
    {
        Function = function;
        Variables = variables;
        Dimensions = variables.Length;
    }
    protected FloatType Evaluate(IDataAccess<FloatType> variables)
    {
        return Function(variables);
    }
    protected void Step(IDataAccess<FloatType> change)
    {
        for (var i = 0; i < Dimensions; i++)
            Variables[i] -= change[i];
    }
    protected void UndoStep(IDataAccess<FloatType> change)
    {
        for (var i = 0; i < Dimensions; i++)
            Variables[i] += change[i];
    }
    /// <summary>
    /// Computes a gradient of error function by nudging it by theta.
    /// </summary>
    protected void ComputeGradient(IDataAccess<FloatType> gradient, FloatType currentEvaluation)
    {
        Parallel.For(0, Dimensions, i =>
        {
            var gradientDataAccess = new GradientDataAccess<FloatType>(Variables, 0, 0);
            gradientDataAccess.Reset(i, Variables[i] + Theta);
            var after = Evaluate(gradientDataAccess);
            gradient[i] = (after - currentEvaluation) / Theta;
        });
    }
    public abstract int Descent(int maxIterations);
}