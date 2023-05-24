using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

public class Sigmoid : IActivationFunction
{
    public Vector Activation(Vector x)
    {
        return (Vector)x.Map(x=>1.0f/(1+MathF.Exp(-x)));
    }

    public Vector ActivationDerivative(Vector x)
    {
        return (Vector)x.Map(x=>1-(1.0f/(1+MathF.Exp(-x))));
    }
}
