using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

public class Tanh : IActivationFunction
{
    public FVector Activation(FVector x)
    {
        return (FVector)x.Map(x=>MathF.Tanh(x));
    }

    public FVector ActivationDerivative(FVector x)
    {
        return (FVector)x.Map(x=>{
            var tanh = MathF.Tanh(x);
            return 1-tanh*tanh;
        });
    }
}
