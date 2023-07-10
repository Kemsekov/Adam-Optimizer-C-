namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

public class Linear : IActivationFunction
{
    public FVector Activation(FVector x)
    {
        return x.Map(x=>x);
    }

    public FVector ActivationDerivative(FVector x)
    {
        return x.Map(x=>1.0f);
    }
}
