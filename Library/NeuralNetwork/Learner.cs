
using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork;

public interface ILearner
{
    void Learn();
    void Unlearn();
}

/// <summary>
/// Normalizes learning rate to be always consistent, so we always step into weights-space
/// by a constant value
/// </summary>
/// <param name="layer"></param>
/// <param name="biasesGradient"></param>
/// <param name="layerInput"></param>
/// <param name="learningRate"></param>
/// <returns></returns>
public record NormalizedLearningRateLearner(ILayer layer, Vector biasesGradient, Vector layerInput, float learningRate) : ILearner
{
    float normalizedLearningRate = float.MaxValue;
    public void Learn()
    {
        var sum = layerInput.Sum();
        var totalWeightsChange = MathF.Sqrt(biasesGradient.Sum(x => x * x * sum * sum));
        if(normalizedLearningRate==float.MaxValue)
            normalizedLearningRate = learningRate / totalWeightsChange;
        for (int k = 0; k < layerInput.Count; k++)
        {
            var kInput = layerInput[k];
            if (kInput == 0) continue;
            for (int j = 0; j < layer.Weights.RowCount; j++)
            {
                var weightGradient = biasesGradient[j] * kInput;
                layer.Weights[j, k] -= normalizedLearningRate * weightGradient;
            }
        }
        layer.Bias.MapIndexedInplace((j, x) => x - learningRate * biasesGradient[j]);
    }

    public void Unlearn()
    {
        var weightsGradient = (int j, int k) => biasesGradient[j] * layerInput[k];

        layer.Weights.MapIndexedInplace((j, k, x) => x + normalizedLearningRate * weightsGradient(j, k));
        layer.Bias.MapIndexedInplace((j, x) => x + learningRate * biasesGradient[j]);
    }
}

/// <summary>
/// Applies backpropagation to model weights in ordinary way
/// </summary>
public record Learner(ILayer layer, Vector biasesGradient, Vector layerInput, float learningRate) : ILearner
{
    public void Learn()
    {
        for (int k = 0; k < layerInput.Count; k++)
        {
            var kInput = layerInput[k];
            if (kInput == 0) continue;
            for (int j = 0; j < layer.Weights.RowCount; j++)
            {
                var weightGradient = biasesGradient[j] * kInput;
                layer.Weights[j, k] -= learningRate * weightGradient;
            }
        }
        layer.Bias.MapIndexedInplace((j, x) => x - learningRate * biasesGradient[j]);

    }
    /// <summary>
    /// Unlearns last learned weights and biases
    /// </summary>
    public void Unlearn()
    {
        var weightsGradient = (int j, int k) => biasesGradient[j] * layerInput[k];
        layer.Weights.MapIndexedInplace((j, k, x) => x + learningRate * weightsGradient(j, k));
        layer.Bias.MapIndexedInplace((j, x) => x + learningRate * biasesGradient[j]);
    }
}
