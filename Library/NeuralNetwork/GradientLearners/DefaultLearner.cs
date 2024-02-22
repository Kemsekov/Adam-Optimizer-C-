namespace GradientDescentSharp.NeuralNetwork;

/// <summary>
/// Applies backpropagation to model weights in ordinary way
/// </summary>
public record DefaultLearner(LearningData LearningData, IRegularization? Regularization = null) : LearnerBase(LearningData)
{
    ///<inheritdoc/>
    public static Func<LearningData, ILearner> Factory(IRegularization? regularization = null)
    {
        return x => new DefaultLearner(x, regularization);
    }

    ///<inheritdoc/>
    public override void Learn()
    {
        var reg = Regularization ?? new NoRegularization();

        //you cannot imagine how much this stuff
        //improves cpu cache hit
        //--------------
        var layerSize = layerInput.Count;
        var rows = layer.Weights.RowCount;
        var weights = layer.Weights;
        var regularization = reg.WeightDerivative;
        
        Func<int, int, float> weightsGet = weights.Storage.At;
        Action<int, int, float> weightsSet = weights.Storage.At;
        Func<int, float> layerGet = layerInput.Storage.At;
        Func<int, float> biasesGradientGet = biasesGradient.Storage.At;
        
        Func<int, float> layerBiasGet = layer.Bias.Storage.At;
        Action<int, float> layerBiasSet = layer.Bias.Storage.At;

        //--------------

        for (int k = 0; k < layerSize; k++)
        {
            var kInput = layerGet(k);
            if (kInput == 0) continue;

            for (int j = 0; j < rows; j++)
            {
                var weightsValue = weightsGet(j, k);
                var weightGradient = biasesGradientGet(j) * kInput + regularization(weightsValue);
                weightsSet(j, k, weightsValue - learningRate * weightGradient);
            }
        }
        for (int j = 0; j < rows; j++){
            var v = layerBiasGet(j);
            layerBiasSet(j,v-learningRate * biasesGradientGet(j));
        }
        // layer.Bias.MapIndexedInplace((j, x) => x - learningRate * biasesGradientGet(j));
    }
    /// <summary>
    /// Unlearns last learned weights and biases
    /// </summary>
    public override void Unlearn()
    {
        if (Regularization is not null)
        {
            throw new NotSupportedException("Unlearn supported only on learners without regularization");
        }
        var weightsGradient = (int j, int k) => biasesGradient[j] * layerInput[k];
        layer.Weights.MapIndexedInplace((j, k, x) => x + learningRate * weightsGradient(j, k));
        layer.Bias.MapIndexedInplace((j, x) => x + learningRate * biasesGradient[j]);
    }
}
