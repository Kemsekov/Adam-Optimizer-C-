using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

public class Softmax : IActivationFunction
{
    public Vector Activation(Vector x)
    {
        var sum = x.Sum(x=>MathF.Exp(x));
        return (Vector)x.Map(x=>MathF.Exp(x)/sum);
    }

    public Vector ActivationDerivative(Vector x)
    {
        var sum = x.Sum(x=>MathF.Exp(x));
        return (Vector)x.Map(x=>{
            var ex = MathF.Exp(x);
            var variable = ex/sum;
            return variable-variable*variable;
        });
    }
}
