namespace GradientDescentSharp.NeuralNetwork.WeightInitializers;
/// <summary>
/// No weight initialization implementation. <br/>
/// Layer creation requires some weight initializer with it, use
/// this one when you don't want to initialize weights at all
/// </summary>
public class NoInitialization : IWeightsInit
{
    ///<inheritdoc/>
    public void InitBiasWeights(FTensor bias)
    {
    }

    ///<inheritdoc/>
    public void InitWeights(FTensor layer)
    {
    }

    ///<inheritdoc/>
    public float SampleWeight(FTensor layer)
    {
        return 0;
    }
}
