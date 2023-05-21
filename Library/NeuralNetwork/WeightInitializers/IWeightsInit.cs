using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork.WeightInitializers;

public interface IWeightsInit{
    public void InitWeights(Vector bias);
    public void InitWeights(Matrix layer);
    /// <summary>
    /// Get a single sample that could be assigned to weights in given layer weights
    /// </summary>
    public float SampleWeight(Matrix layer);
}