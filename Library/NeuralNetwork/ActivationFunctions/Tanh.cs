using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

public class Tanh : IActivationFunction
{
    public Vector Activation(Vector x)
    {
        return (Vector)x.Map(x=>MathF.Tanh(x));
    }

    public Vector ActivationDerivative(Vector x)
    {
        return (Vector)x.Map(x=>{
            var tanh = MathF.Tanh(x);
            return 1-tanh*tanh;
        });
    }
}
