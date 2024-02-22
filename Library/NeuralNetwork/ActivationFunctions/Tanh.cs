namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

///<inheritdoc/>
public class Tanh : IActivationFunction
{
    ///<inheritdoc/>
    public IWeightsInit WeightsInit { get; set; } = new GlorotUniform();
    ///<inheritdoc/>
    public FTensor Activation(FTensor x)
    {
        return x.Map(x => MathF.Tanh(x));
    }

    ///<inheritdoc/>
    public FTensor ActivationDerivative(FTensor x)
    {
        return x.Map(x =>
        {
            var tanh = MathF.Tanh(x);
            return 1 - tanh * tanh;
        });
    }
}
