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
    public unsafe override void Learn()
    {
        var reg = Regularization ?? new NoRegularization();
        var bgSpan = biasesGradient.AsSpan();
        var liSpan = layerInput.AsSpan();
        //optimize it beyond reason
        layer.Weights.MapInplace(bgSpan, liSpan, (index, bg, li, weight) =>
        {
            var j = index[0];
            var k = index[1];
            var weightGradient = bg[j] * li[k] + reg.WeightDerivative(weight);
            return weight - learningRate * weightGradient;
        });
        layer.Bias.VecMapInplace(bgSpan,(j,s, x) => x - learningRate * s[j]);
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
        var weightsGradient = (int[] jk) => biasesGradient[jk[0]] * layerInput[jk[1]];
        layer.Weights.MapInplace((ind, x) => x + learningRate * weightsGradient(ind));
        layer.Bias.VecMapInplace((j, x) => x + learningRate * biasesGradient[j]);
    }
}
