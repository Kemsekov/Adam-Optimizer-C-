namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

///<inheritdoc/>
public class Sigmoid : IActivationFunction
{
    ///<inheritdoc/>
    public IWeightsInit WeightsInit { get; set; } = new GlorotUniform();
    ///<inheritdoc/>
    public FTensor Activation(FTensor x)
    {
        return x.Map(x => 1.0f / (1.0f + MathF.Exp(-x)));
    }

    ///<inheritdoc/>
    public FTensor ActivationDerivative(FTensor x)
    {
        return x.Map(x => 1.0f - (1.0f / (1.0f + MathF.Exp(-x))));
    }
}
