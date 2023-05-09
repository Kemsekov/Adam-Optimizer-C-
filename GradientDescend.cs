namespace AdamOptimizer;

/// <summary>
/// Contains gradient descend implementations.
/// </summary>
public class GradientDescend
{
    public Func<double[], double> Function { get; }
    public double[] Variables { get; }
    public int Dimensions { get; }
    public double Beta1=0.9;
    public double Beta2=0.99;
    public double Epsilon=1e-6;
    public GradientDescend(double[] variables, Func<double[], double> function)
    {
        Function = function;
        Variables = variables;
        Dimensions = variables.Length;
    }
    double Evaluate(double[] variables)
    {
        return Function(variables);
    }
    void ComputeGradient(double[] gradient,double currentEvaluation){
        if (currentEvaluation == 0) return;
        for (int i = 0; i < Dimensions; i++)
        {
            Variables[i] += Epsilon;
            var after = Evaluate(Variables);
            Variables[i] -= Epsilon;
            gradient[i] = (after - currentEvaluation)/Epsilon;
        }
    }
    void ComputeChangeMine(double[] change, double learningRate, double currentEvaluation){
        ComputeGradient(change,currentEvaluation);
        var length = Math.Sqrt(change.Sum(x=>x*x));
        var coefficient = learningRate/length;
        for(int i = 0;i<Dimensions;i++){
            change[i]*=coefficient;
        }
    }
    void ComputeChangeAdam(double[] change, double learningRate,double currentEvaluation)
    {
        ComputeGradient(change,currentEvaluation);
        var gradient = change;
        int t = 0; // The timestep counter

        // Initialize the first and second moment vector

        // Loop over the parameters
        double previousFirstMomentum = 0;
        double previousSecondMomentum = 0;
        double firstMomentum;
        double secondMomentum;
        for (int i = 0; i < Dimensions; i++)
        {
            // Increment the timestep counter
            t++;

            // Update the biased first moment estimate
            firstMomentum = Beta1 * previousFirstMomentum + (1 - Beta1) * gradient[i];

            // Update the biased second moment estimate
            secondMomentum = Beta2 * previousSecondMomentum + (1 - Beta2) * gradient[i] * gradient[i];

            // Compute the bias-corrected first moment estimate
            double m_hat = firstMomentum / (1 - Math.Pow(Beta1, t));

            // Compute the bias-corrected second moment estimate
            double v_hat = secondMomentum / (1 - Math.Pow(Beta2, t));
            
            previousFirstMomentum = firstMomentum;
            previousSecondMomentum = secondMomentum;
            // Update the parameters    
            change[i] = learningRate * m_hat / (Math.Sqrt(v_hat) + Epsilon);
        }
    }
    void Step(double[] change)
    {
        for (var i = 0; i < Dimensions; i++)
        {
            var c = change[i];
            if (c == 0) continue;
            Variables[i] -= c;
        }
    }
    void UndoStep(double[] change)
    {
        for (var i = 0; i < Dimensions; i++)
        {
            var c = change[i];
            if (c == 0) continue;
            Variables[i] += c;
        }
    }
    public int MineDescent(int maxIterations, double learningRate = 1, double theta = 0.001){
        var change = new double[Dimensions];
        var iterations = 0;
        while (maxIterations-- > 0)
        {
            iterations++;
            var beforeStep = Evaluate(Variables);
            ComputeChangeMine(change, learningRate,beforeStep);
            Step(change);
            var afterStep = Evaluate(Variables);
            var diff = Math.Abs(afterStep - beforeStep);
            if (diff <= theta) break;
            if (afterStep >= beforeStep)
            {
                UndoStep(change);
                learningRate*=1-Beta1;
            }
        }
        return iterations;
    }
    public int AdamDescent(int maxIterations, double learningRate = 1, double theta = 0.001)
    {
        var change = new double[Dimensions];
        var iterations = 0;
        while (maxIterations-- > 0)
        {
            iterations++;
            var beforeStep = Evaluate(Variables);
            ComputeChangeAdam(change, learningRate,beforeStep);
            Step(change);
            var afterStep = Evaluate(Variables);
            var diff = Math.Abs(afterStep - beforeStep);
            if (diff <= theta) break;
            if (afterStep >= beforeStep)
            {
                UndoStep(change);
                learningRate*=1-Beta1;
            }
        }
        return iterations;
    }
}