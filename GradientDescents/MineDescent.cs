namespace GradientDescentSharp.GradientDescents;
public class MineDescent : GradientDescentBase
{
    /// <summary>
    /// Descent process can be logged here
    /// </summary>
    public ILogger? Logger;
    public double DescentRateDecreaseRate = 0.1;
    public MineDescent(IDataAccess<double> variables, Func<IDataAccess<double>, double> function) : base(variables, function)
    {
    }
    void ComputeChangeMine(IDataAccess<double> change, double learningRate, double currentEvaluation)
    {
        ComputeGradient(change, currentEvaluation);
        var length = Math.Sqrt(change.Sum(x => x * x));
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
            if (afterStep >= beforeStep)
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