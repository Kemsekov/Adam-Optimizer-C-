using System.Numerics;

namespace GradientDescentSharp.GradientDescents;
/// <summary>
/// Custom gradient descent implementation made my Kemsekov. <br/>
/// In around 80% cases works better than adam optimizer<br/>
/// Have similar logic under the hood, whenever we hit a worse error function value
/// than before, we rollback, decrease learning rate and continue.
/// </summary>
public abstract class MineDescent<TFloat> : ReversibleLoggedGradientDescentBase<TFloat>
where TFloat : unmanaged, INumber<TFloat>
{
    /// <summary>
    /// How much decrease descent rate when we step into bigger error value.<br/>
    /// By default it is 0.1, so when we step into worse error function value,
    /// we will divide learning rate by 10.
    /// </summary>
    public override TFloat DescentRateDecreaseRate{get;set;} = (TFloat)(0.1 as dynamic);
    /// <summary>
    /// Creates new instance of <see cref="MineDescent"/> 
    /// </summary>
    /// <inheritdoc/>
    public MineDescent(IDataAccess<TFloat> variables, Func<IDataAccess<TFloat>, TFloat> function) : base(variables, function)
    {
    }
    /// <summary>
    /// Computes length of vector <paramref name="change"/> 
    /// </summary>
    protected abstract TFloat Length(IDataAccess<TFloat> change);
    ///<inheritdoc/>
    protected override void ComputeChange(IDataAccess<TFloat> change, TFloat learningRate, TFloat currentEvaluation)
    {
        ComputeGradient(change, Variables, currentEvaluation);
        var length = Length(change);
        if (length == TFloat.Zero)
        {
            return;
        }
        var coefficient = learningRate / length;
        for (int i = 0; i < Dimensions; i++)
        {
            change[i] *= coefficient;
        }
    }
}
///<inheritdoc/>
public class MineDescent : MineDescent<double>
{
    ///<inheritdoc/>
    public MineDescent(IDataAccess<double> variables, Func<IDataAccess<double>, double> function) : base(variables, function)
    {
    }

    ///<inheritdoc/>
    protected override double Length(IDataAccess<double> change) => Math.Sqrt(change.Sum(x => x * x));
}

///<inheritdoc/>
public class MineDescentSingle : MineDescent<float>
{
    ///<inheritdoc/>
    public MineDescentSingle(IDataAccess<float> variables, Func<IDataAccess<float>, float> function) : base(variables, function)
    {
    }

    ///<inheritdoc/>
    protected override float Length(IDataAccess<float> change) => MathF.Sqrt(change.Sum(x => x * x));
}
