using System.Numerics;

namespace GradientDescentSharp.GradientDescents;
/// <summary>
/// Custom gradient descent implementation made my Kemsekov. <br/>
/// In around 80% cases works better than adam optimizer<br/>
/// Have similar logic under the hood, whenever we hit a worse error function value
/// than before, we rollback, decrease learning rate and continue.
/// </summary>
public abstract class MineDescent<TFloat> : GradientDescentBase<TFloat>
where TFloat : unmanaged, INumber<TFloat>
{
    /// <summary>
    /// Descent process can be logged here
    /// </summary>
    public ILogger? Logger;
    /// <summary>
    /// How much decrease descent rate when we step into bigger error value.<br/>
    /// By default it is 0.1, so when we step into worse error function value,
    /// we will divide  learning rate by 10.
    /// </summary>
    public TFloat DescentRateDecreaseRate = TFloat.One*(0.1 as dynamic);
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
    void ComputeChangeMine(IDataAccess<TFloat> change, TFloat learningRate, TFloat currentEvaluation)
    {
        ComputeGradient(change, Variables, currentEvaluation);
        var length = Length(change);
        if (length == TFloat.Zero)
        {
            Logger?.LogLine("Found exact minima!");
            return;
        }
        var coefficient = learningRate / length;
        for (int i = 0; i < Dimensions; i++)
        {
            change[i] *= coefficient;
        }
    }
    ///<inheritdoc/>
    public override int Descent(int maxIterations)
    {
        Logger?.LogLine("--------------Mine descent began");

        using RentedArrayDataAccess<TFloat> change = new(ArrayPoolStorage.RentArray<TFloat>(Dimensions));
        var iterations = 0;
        var descentRate = DescentRate;
        var beforeStep = Evaluate(Variables);
        while (iterations++ < maxIterations)
        {
            ComputeChangeMine(change, descentRate, beforeStep);
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
        }
        Logger?.LogLine($"--------------Mine done in {iterations} iterations");

        return iterations;
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
