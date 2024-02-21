namespace GradientDescentSharp.NeuralNetwork;

/// <summary>
/// Normalizes learning rate to be proportional to gradient length, so we always learn weights
/// by a constant value. <br/>
/// Inefficient in a case, where gradient values is big, because it will make 
/// model converge in a slower rate. <br/>
/// Efficient when gradient values is small, because in default learning scheme 
/// it will slow learning rate, but this one will not slow down.<br/>
/// You may imagine it as if we are rolling down from a mountain and riding on almost flat
/// surface all in the same fixated speed.
/// </summary>
public record NormalizedLearner(LearningData LearningData,IRegularization? Regularization = null) : LearnerBase(LearningData)
{
    ///<inheritdoc/>
    public static Func<LearningData, ILearner> Factory(IRegularization? regularization = null)
    {
        return x=>new NormalizedLearner(x,regularization);
    }
    float normalizedLearningRate = float.MaxValue;
    ///<inheritdoc/>
    public override void Learn()
    {
        var reg = Regularization ?? new NoRegularization();
        // Because it is expensive to compute this value, I decided to make it in one-initialization
        if(normalizedLearningRate==float.MaxValue){
            var sum = layerInput.Sum();
            var totalWeightsChange = MathF.Sqrt(biasesGradient.Sum(x => x * x * sum * sum));
            normalizedLearningRate = learningRate / totalWeightsChange;
        }

        var biasLearningRate = (float)(learningRate/biasesGradient.L2Norm());    

        for (int k = 0; k < layerInput.Count; k++)
        {
            var kInput = layerInput[k];
            if (kInput == 0) continue;
            for (int j = 0; j < layer.Weights.RowCount; j++)
            {
                var weightGradient = biasesGradient[j] * kInput + reg.WeightDerivative(layer.Weights[j, k]);
                layer.Weights[j, k] -= normalizedLearningRate * weightGradient;
            }
        }
        layer.Bias.MapIndexedInplace((j, x) => x - biasLearningRate * biasesGradient[j]);
    }

    ///<inheritdoc/>
    public override void Unlearn()
    {
        var weightsGradient = (int j, int k) => biasesGradient[j] * layerInput[k];
        var biasLearningRate = (float)(learningRate/biasesGradient.L2Norm());    

        layer.Weights.MapIndexedInplace((j, k, x) => x + normalizedLearningRate * weightsGradient(j, k));
        layer.Bias.MapIndexedInplace((j, x) => x + biasLearningRate * biasesGradient[j]);
    }
}
