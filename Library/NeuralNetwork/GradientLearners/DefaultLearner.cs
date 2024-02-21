namespace GradientDescentSharp.NeuralNetwork;

/// <summary>
/// Applies backpropagation to model weights in ordinary way
/// </summary>
public record DefaultLearner(LearningData LearningData, IRegularization? Regularization = null) : LearnerBase(LearningData)
{
    ///<inheritdoc/>
    public static Func<LearningData, ILearner> Factory(IRegularization? regularization = null)
    {
        return x => new DefaultLearner(x,regularization);
    }

    ///<inheritdoc/>
    public override void Learn()
    {
        var reg = Regularization ?? new NoRegularization();
        var layerSize = layerInput.Count;
        var rows = layer.Weights.RowCount;
        var weights = layer.Weights;
        var regularization = reg.WeightDerivative;

        for (int k = 0; k < layerSize; k++)
        {
            var kInput = layerInput[k];
            if (kInput == 0) continue;

            for (int j = 0; j < rows; j++)
            {
                var weightGradient = biasesGradient[j] * kInput + regularization(weights[j, k]);
                weights[j, k] -= learningRate * weightGradient;
            }
        }
        layer.Bias.MapIndexedInplace((j, x) => x - learningRate * biasesGradient[j]);
    }

    /// <summary>
    /// Unlearns last learned weights and biases
    /// </summary>
    public override void Unlearn()
    {
        if(Regularization is not null){
            throw new NotSupportedException("Unlearn supported only on learners without regularization");
        }
        var weightsGradient = (int j, int k) => biasesGradient[j] * layerInput[k];
        layer.Weights.MapIndexedInplace((j, k, x) => x + learningRate * weightsGradient(j, k));
        layer.Bias.MapIndexedInplace((j, x) => x + learningRate * biasesGradient[j]);
    }
}
