namespace GradientDescentSharp.NeuralNetwork.WeightInitializers;
/// <summary>
/// No weight initialization implementation. <br/>
/// Layer creation requires some weight initializer with it, use
/// this one when you don't want to initialize weights at all
/// </summary>
public class NoInitialization : IWeightsInit
{
    public void InitWeights(FVector bias)
    {
    }

    public void InitWeights(FMatrix layer)
    {
    }

    public float SampleWeight(FMatrix layer)
    {
        return 0;
    }
}
