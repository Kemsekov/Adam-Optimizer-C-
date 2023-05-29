namespace GradientDescentSharp.GradientDescents;
/// <summary>
/// Custom gradient descent implementation made my Kemsekov. <br/>
/// In around 80% cases works better than adam optimizer<br/>
/// Have similar logic under the hood, whenever we hit a worse error function value
/// than before, we rollback, decrease learning rate and continue.
/// </summary>
public class MineDescent : GradientDescentBase
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
    public double DescentRateDecreaseRate = 0.1;
    public MineDescent(IDataAccess<double> variables, Func<IDataAccess<double>, double> function) : base(variables, function)
    {
    }
    void ComputeChangeMine(IDataAccess<double> change, double learningRate, double currentEvaluation)
    {
        ComputeGradient(change, currentEvaluation);
        var length = Math.Sqrt(change.Sum(x => x * x));
        if (length == 0)
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
    public override int Descent(int maxIterations)
    {
        Logger?.LogLine("--------------Mine descent began");

        using RentedArrayDataAccess<double> change = new(ArrayPoolStorage.RentArray<double>(Dimensions));
        var iterations = 0;
        var descentRate = DescentRate;
        var beforeStep = Evaluate(Variables);
        while (iterations++ < maxIterations)
        {
            ComputeChangeMine(change, descentRate, beforeStep);
            Step(change);
            var afterStep = Evaluate(Variables);
            var diff = Math.Abs(afterStep - beforeStep);
            Logger?.LogLine($"Error is {afterStep}");
            Logger?.LogLine($"Changed by {diff}");
            if (diff <= Theta) break;
            if (afterStep >= beforeStep || double.IsNaN(afterStep))
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
