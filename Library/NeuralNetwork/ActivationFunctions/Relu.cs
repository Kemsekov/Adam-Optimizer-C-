using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

public class Relu : IActivationFunction
{
    public Vector Activation(Vector x)
    {
        return (Vector)x.Map(x=>Math.Max(0,x));
    }

    public Vector ActivationDerivative(Vector x)
    {
        return (Vector)x.Map(x=>x>0 ? 1.0f : 0.0f);
    }
}
