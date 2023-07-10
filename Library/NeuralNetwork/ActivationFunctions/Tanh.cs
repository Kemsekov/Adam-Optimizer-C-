namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

public class Tanh : IActivationFunction
{
    public FVector Activation(FVector x)
    {
        return x.Map(x=>MathF.Tanh(x));
    }

    public FVector ActivationDerivative(FVector x)
    {
        return x.Map(x=>{
            var tanh = MathF.Tanh(x);
            return 1-tanh*tanh;
        });
    }
}
