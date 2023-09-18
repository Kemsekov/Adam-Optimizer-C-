using System.Numerics;

namespace GradientDescentSharp.GradientDescents;
/// <summary>
/// Base class for gradient descent implementations
/// </summary>
public abstract class GradientDescentBase<TFloat> : IGradientDescent
where TFloat : unmanaged, INumber<TFloat>
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
    public Action<IDataAccess<TFloat>, IDataAccess<TFloat>, TFloat> ComputeGradient { get; set; }
    /// <summary>
    /// Error function that we need to minimize. Returns values >= 0
    /// </summary>
    /// <value></value>
    public Func<IDataAccess<TFloat>, TFloat> Function { get; }
    /// <summary>
    /// Function variables that need to be adjusted to minimize error function 
    /// </summary>
    public IDataAccess<TFloat> Variables { get; }
    /// <summary>
    /// Length of function variables
    /// </summary>
    public int Dimensions { get; }

    /// <summary>
    /// Descent rate. The higher, the more.
    /// </summary>
    public TFloat DescentRate;// = 0.05;
    /// <summary>
    /// If gradient descent step error function value changes less than this value, it means
    /// we need to step descending. <br/>
    /// In short terms, this value defines how precise our find local minima need to be.<br/>
    /// Set to zero, if you need to find the most precise possible solution
    /// </summary>
    public TFloat Theta;// = 0.0001;
    /// <summary>
    /// This value is a derivative epsilon, that used to compute gradient
    /// </summary>
    public TFloat Epsilon;// = 0.0001;
    /// <summary>
    /// Current iteration number
    /// </summary>
    protected int Iteration;
    /// <summary>
    /// Create new instance of gradient descent
    /// </summary>
    /// <param name="variables">Variables that will be adjusted</param>
    /// <param name="function">
    /// Loss function.<br/>
    /// Good rule of thumb: return MSE as error function<br/>
    /// So if your error values is a,b,c,d => then return error as: a^2+b^2+c^2+d^2
    /// </param>
    public GradientDescentBase(IDataAccess<TFloat> variables, Func<IDataAccess<TFloat>, TFloat> function)
    {
        DescentRate = (TFloat)(0.05 as dynamic);
        Theta = (TFloat)(0.0001 as dynamic);
        Epsilon = (TFloat)(0.0001 as dynamic);
        Function = function;
        Variables = variables;
        Dimensions = variables.Length;
        ComputeGradient = ComputeGradientDefaultImplementation;
    }
    /// <summary>
    /// Computes variables change
    /// </summary>
    protected abstract void ComputeChange(IDataAccess<TFloat> change, TFloat learningRate, TFloat currentEvaluation);
    /// <summary>
    /// Evaluate function that we try to minimize at given variables
    /// </summary>
    public TFloat Evaluate(IDataAccess<TFloat> variables)
    {
        return Function(variables);
    }
    /// <summary>
    /// Applies change to current variables
    /// </summary>
    protected void Step(IDataAccess<TFloat> change)
    {
        for (var i = 0; i < Dimensions; i++)
            Variables[i] -= change[i];
    }
    /// <summary>
    /// Undo apply of change
    /// </summary>
    protected void UndoStep(IDataAccess<TFloat> change)
    {
        for (var i = 0; i < Dimensions; i++)
            Variables[i] += change[i];
    }
    /// <summary>
    /// Computes a gradient of error function by nudging it by theta.
    /// </summary>
    protected void ComputeGradientDefaultImplementation(IDataAccess<TFloat> gradient, IDataAccess<TFloat> variables, TFloat currentEvaluation)
    {
        Parallel.For(0, Dimensions, i =>
        {
            var gradientDataAccess = new GradientDataAccess<TFloat>(variables, i, variables[i] + Epsilon);
            var after = Evaluate(gradientDataAccess);
            gradient[i] = (after - currentEvaluation) / Epsilon;
        });
    }
    ///<inheritdoc/>
    public abstract IEnumerable<int> Descent();
    /// <summary>
    /// Does up to maxIterations descent steps
    /// </summary>
    /// <returns>Number of descent steps</returns>
    public int Descent(int maxIterations){
        Descent().Take(maxIterations).ToArray();
        return Iteration;
    }
}

/// <summary>
/// Reversible descent impl.
/// </summary>
/// <typeparam name="TFloat"></typeparam>
public abstract class ReversibleLoggedGradientDescentBase<TFloat> : GradientDescentBase<TFloat>
where TFloat : unmanaged, INumber<TFloat>
{
    /// <summary>
    /// How much decrease descent rate when we step into bigger error value.<br/>
    /// If it is 0.1, then when we step into worse error function value,
    /// we will divide learning rate by 10.
    /// </summary>
    public abstract TFloat DescentRateDecreaseRate { get;set; }
    /// <summary>
    /// Descent process can be logged here
    /// </summary>
    public ILogger? Logger;
    ///<inheritdoc/>
    protected ReversibleLoggedGradientDescentBase(IDataAccess<TFloat> variables, Func<IDataAccess<TFloat>, TFloat> function) : base(variables, function)
    {
    }

    ///<inheritdoc/>
    public override IEnumerable<int> Descent()
    {
        Logger?.LogLine("--------------Mine descent began");
        Iteration = 0;
        using RentedArrayDataAccess<TFloat> change = new(ArrayPoolStorage.RentArray<TFloat>(Dimensions));
        var descentRate = DescentRate;
        var beforeStep = Evaluate(Variables);
        while (true)
        {
            Iteration++;
            ComputeChange(change, descentRate, beforeStep);
            Step(change);
            var afterStep = Evaluate(Variables);
            var diff = Math<TFloat>.Abs(afterStep - beforeStep);
            Logger?.LogLine($"Error is {afterStep}");
            Logger?.LogLine($"Changed by {diff}");
            if (diff <= Theta) break;
            if (afterStep >= beforeStep || TFloat.IsNaN(afterStep))
            {
                Logger?.LogLine($"Undo step. Decreasing descentRate.");
                UndoStep(change);
                descentRate *= DescentRateDecreaseRate;
            }
            else
            {
                beforeStep = afterStep;
            }
            Logger?.LogLine($"-------------");
            yield return Iteration;
        }
        Logger?.LogLine($"--------------Mine done in {Iteration} iterations");
    }
}