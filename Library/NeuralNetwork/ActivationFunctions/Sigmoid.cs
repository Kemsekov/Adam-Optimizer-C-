namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

///<inheritdoc/>
public class Sigmoid : IActivationFunction
{
    ///<inheritdoc/>
    public IWeightsInit WeightsInit { get; set; } = new GlorotUniform();
    ///<inheritdoc/>
    public FVector Activation(FVector x)
    {
        return x.Map(x => 1.0f / (1 + MathF.Exp(-x)));
    }

    ///<inheritdoc/>
    public FVector ActivationDerivative(FVector x)
    {
        return x.Map(x => 1 - (1.0f / (1 + MathF.Exp(-x))));
    }
}
