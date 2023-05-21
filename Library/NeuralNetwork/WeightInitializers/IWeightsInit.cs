using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork.WeightInitializers;

public interface IWeightsInit{
    public void InitWeights(Vector bias);
    public void InitWeights(Matrix layer);
}