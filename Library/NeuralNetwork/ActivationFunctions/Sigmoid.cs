namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

public class Sigmoid : IActivationFunction
{
    public FVector Activation(FVector x)
    {
        return x.Map(x=>1.0f/(1+MathF.Exp(-x)));
    }

    public FVector ActivationDerivative(FVector x)
    {
        return x.Map(x=>1-(1.0f/(1+MathF.Exp(-x))));
    }
}
