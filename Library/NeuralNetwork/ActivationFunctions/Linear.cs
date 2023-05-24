using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

public class Linear : IActivationFunction
{
    public Vector Activation(Vector x)
    {
        return (Vector)x.Map(x=>x);
    }

    public Vector ActivationDerivative(Vector x)
    {
        return (Vector)x.Map(x=>1.0f);
    }
}
