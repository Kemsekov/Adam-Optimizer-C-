namespace GradientDescentSharp.GradientDescents;

public class AdamDescent : GradientDescentBase
{
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
            change[i] = learningRate * m_hat / (Math.Sqrt(v_hat) + Theta);
        });
    }
    public override int Descent(int maxIterations)
    {
        using RentedArrayDataAccess<double> change = new(ArrayPoolStorage.RentArray<double>(Dimensions));
        var iterations = 0;
        var descentRate = DescentRate;
        while (iterations++<maxIterations)
        {
            var beforeStep = Evaluate(Variables);
            ComputeChangeAdam(change, descentRate, beforeStep,iterations);
            Step(change);
            var afterStep = Evaluate(Variables);
            var diff = Math.Abs(afterStep - beforeStep);
            if (diff <= Theta) break;    
            if (afterStep >= beforeStep)
            {
                UndoStep(change);
                descentRate *= 1 - Beta1;
            }
        }
        return iterations;
    }
}