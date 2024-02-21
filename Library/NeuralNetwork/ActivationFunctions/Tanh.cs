namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

///<inheritdoc/>
public class Tanh : IActivationFunction
{
    ///<inheritdoc/>
    public IWeightsInit WeightsInit { get; set; } = new GlorotUniform();
    ///<inheritdoc/>
    public FVector Activation(FVector x)
    {
        return x.Map(x => MathF.Tanh(x));
    }

    ///<inheritdoc/>
    public FVector ActivationDerivative(FVector x)
    {
        return x.Map(x =>
        {
            var tanh = MathF.Tanh(x);
            return 1 - tanh * tanh;
        });
    }
}
