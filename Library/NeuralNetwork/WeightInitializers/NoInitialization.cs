namespace GradientDescentSharp.NeuralNetwork.WeightInitializers;
/// <summary>
/// No weight initialization implementation. <br/>
/// Layer creation requires some weight initializer with it, use
/// this one when you don't want to initialize weights at all
/// </summary>
public class NoInitialization : IWeightsInit
{
    ///<inheritdoc/>
    public void InitWeights(FVector bias)
    {
    }

    ///<inheritdoc/>
    public void InitWeights(FMatrix layer)
    {
    }

    ///<inheritdoc/>
    public float SampleWeight(FMatrix layer)
    {
        return 0;
    }
}
