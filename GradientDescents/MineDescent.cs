namespace GradientDescentSharp.GradientDescents;
public class MineDescent : GradientDescentBase
{
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
        using RentedArrayDataAccess<double> change = new(ArrayPoolStorage.RentArray<double>(Dimensions));
        var iterations = 0;
        var descentRate = DescentRate;
        while (iterations++ < maxIterations)
        {
            var beforeStep = Evaluate(Variables);
            ComputeChangeMine(change, descentRate, beforeStep);
            Step(change);
            var afterStep = Evaluate(Variables);
            var diff = Math.Abs(afterStep - beforeStep);
            if (diff <= Theta) break;
            if (afterStep >= beforeStep)
            {
                UndoStep(change);
                descentRate *= DescentRateDecreaseRate;
            }
        }
        return iterations;
    }
}