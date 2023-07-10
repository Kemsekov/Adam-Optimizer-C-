namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

public class Relu : IActivationFunction
{
    public FVector Activation(FVector x)
    {
        return x.Map(x=>Math.Max(0,x));
    }

    public FVector ActivationDerivative(FVector x)
    {
        return x.Map(x=>x>0 ? 1.0f : 0.0f);
    }
}
