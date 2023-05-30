namespace GradientDescentSharp.GradientDescents;

/// <summary>
/// Modified implementation of adam descent.<br/>
/// The sole difference is that whenever the next error value is bigger than previous,
/// the model rollback it's values, reduce the learning rate and step again.<br/>
/// It is guaranteed to find local minima.
/// </summary>
public class AdamDescent : GradientDescentBase
{
    /// <summary>
    /// Descent process can be logged here
    /// </summary>
    public ILogger? Logger;
    RentedArray<double> firstMomentum;
    RentedArray<double> secondMomentum;
    public double Beta1 = 0.9;
    public double Beta2 = 0.99;
    
    public AdamDescent(IDataAccess<double> variables, Func<IDataAccess<double>, double> function) : base(variables, function)
    {
        firstMomentum = ArrayPoolStorage.RentArray<double>(Dimensions);
        secondMomentum = ArrayPoolStorage.RentArray<double>(Dimensions);
    }
    void ComputeChangeAdam(IDataAccess<double> change, double learningRate, double currentEvaluation, int iteration)
    {
        ComputeGradient(change, currentEvaluation);
        var gradient = change;
        int t = iteration; // The timestep counter

        // Loop over the parameters
        Parallel.For(0,Dimensions,i=>
        {
            // Update the biased first moment estimate
            firstMomentum[i] = Beta1 * firstMomentum[i] + (1 - Beta1) * gradient[i];

            // Update the biased second moment estimate
            secondMomentum[i] = Beta2 * secondMomentum[i] + (1 - Beta2) * gradient[i] * gradient[i];

            // Compute the bias-corrected first moment estimate
            double m_hat = firstMomentum[i] / (1 - Math.Pow(Beta1, t));

            // Compute the bias-corrected second moment estimate
            double v_hat = secondMomentum[i] / (1 - Math.Pow(Beta2, t));

            // Update the parameters    
            change[i] = learningRate * m_hat / (Math.Sqrt(v_hat) + Epsilon);
        });
    }
    public override int Descent(int maxIterations)
    {
        Logger?.LogLine("--------------Adam descent began");
        using RentedArrayDataAccess<double> change = new(ArrayPoolStorage.RentArray<double>(Dimensions));
        var iterations = 0;
        var descentRate = DescentRate;
        var beforeStep = Evaluate(Variables);
        while (iterations++<maxIterations)
        {
            ComputeChangeAdam(change, descentRate, beforeStep,iterations);
            Step(change);
            var afterStep = Evaluate(Variables);
            var diff = Math.Abs(afterStep - beforeStep);
            Logger?.LogLine($"Error is {afterStep}");
            Logger?.LogLine($"Changed by {diff}");
            if (diff <= Theta) break;    
            if (afterStep >= beforeStep)
            {
                Logger?.LogLine($"Undo step, decreasing descentRate.");
                UndoStep(change);
                descentRate *= 1 - Beta1;
            }
            else{
                beforeStep = afterStep;
            }
            Logger?.LogLine($"-------------");
        }
        Logger?.LogLine($"--------------Adam done in {iterations} iterations");
        return iterations;
    }
}