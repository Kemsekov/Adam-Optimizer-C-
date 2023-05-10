using RentedArraySharp;

namespace AdamOptimizer;

/// <summary>
/// Contains gradient descent implementations.
/// </summary>
public class GradientDescent
{
    public Func<IDataAccess<double>, double> Function { get; }
    public IDataAccess<double> Variables { get; }
    public int Dimensions { get; }
    public double Beta1 = 0.9;
    public double Beta2 = 0.99;
    public double Epsilon = 1e-6;
    public double LearningRate;
    public double Theta;

    public GradientDescent(IDataAccess<double> variables, Func<IDataAccess<double>, double> function)
    {
        Function = function;
        Variables = variables;
        Dimensions = variables.Length;
        firstMomentum = new double[Dimensions];
        secondMomentum = new double[Dimensions];
    }
    double Evaluate(IDataAccess<double> variables)
    {
        return Function(variables);
    }
    void ComputeGradient(IDataAccess<double> gradient, double currentEvaluation)
    {
        Parallel.For(0,Dimensions,i=>{
            var gradientDataAccess = new GradientDataAccess<double>(Variables,0,0);
            gradientDataAccess.Reset(i,Variables[i]+Epsilon);
            var after = Evaluate(gradientDataAccess);
            gradient[i] = (after - currentEvaluation) / Epsilon;
        });
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
    double[] firstMomentum;
    double[] secondMomentum;
    void ComputeChangeAdam(IDataAccess<double> change, double learningRate, double currentEvaluation, int iteration)
    {
        ComputeGradient(change, currentEvaluation);
        var gradient = change;
        int t = iteration; // The timestep counter

        // Loop over the parameters

        for (int i = 0; i < Dimensions; i++)
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
        }
    }
    void Step(IDataAccess<double> change)
    {
        for (var i = 0; i < Dimensions; i++)
            Variables[i] -= change[i];
    }
    void UndoStep(IDataAccess<double> change)
    {
        for (var i = 0; i < Dimensions; i++)
            Variables[i] += change[i];
    }
    public int MineDescent(int maxIterations)
    {
        using RentedArrayDataAccess<double> change = new(ArrayPoolStorage.RentArray<double>(Dimensions));
        var iterations = 0;
        var learningRate = LearningRate;
        while (iterations++<maxIterations)
        {
            var beforeStep = Evaluate(Variables);
            ComputeChangeMine(change, learningRate, beforeStep);
            Step(change);
            var afterStep = Evaluate(Variables);
            var diff = Math.Abs(afterStep - beforeStep);
            if (diff <= Theta) break;
            if (afterStep >= beforeStep)
            {
                UndoStep(change);
                learningRate *= 1 - Beta1;
            }
        }
        return iterations;
    }
    public int AdamDescent(int maxIterations)
    {
        using RentedArrayDataAccess<double> change = new(ArrayPoolStorage.RentArray<double>(Dimensions));
        var iterations = 0;
        var learningRate = LearningRate;
        while (iterations++<maxIterations)
        {
            var beforeStep = Evaluate(Variables);
            ComputeChangeAdam(change, learningRate, beforeStep,iterations);
            Step(change);
            var afterStep = Evaluate(Variables);
            var diff = Math.Abs(afterStep - beforeStep);
            // if (diff <= Theta) break;    
            if (afterStep >= beforeStep)
            {
                UndoStep(change);
                learningRate *= 1 - Beta1;
            }
        }
        return iterations;
    }
}