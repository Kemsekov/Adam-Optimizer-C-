namespace GradientDescentSharp.NeuralNetwork.WeightInitializers;
using MathNet.Numerics.LinearAlgebra.Single;

/// <summary>
/// No weight initialization implementation. <br/>
/// Layer creation requires some weight initializer with it, use
/// this one when you don't want to initialize weights at all
/// </summary>
public class NoInitialization : IWeightsInit
{
    public void InitWeights(Vector bias)
    {
    }

    public void InitWeights(Matrix layer)
    {
    }

    public float SampleWeight(Matrix layer)
    {
        return 0;
    }
}
