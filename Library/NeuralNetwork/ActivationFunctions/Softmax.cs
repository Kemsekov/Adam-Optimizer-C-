using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

public class Softmax : IActivationFunction
{
    public FVector Activation(FVector x)
    {
        var sum = x.Sum(x=>MathF.Exp(x));
        return (FVector)x.Map(x=>MathF.Exp(x)/sum);
    }

    public FVector ActivationDerivative(FVector x)
    {
        var sum = x.Sum(x=>MathF.Exp(x));
        return (FVector)x.Map(x=>{
            var ex = MathF.Exp(x);
            var variable = ex/sum;
            return variable-variable*variable;
        });
    }
}
