using Tensornet;

namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

///<inheritdoc/>
public class Relu : IActivationFunction
{
    ///<inheritdoc/>
    public IWeightsInit WeightsInit { get; set; } = new He2Normal();
    ///<inheritdoc/>
    public FTensor Activation(FTensor x)
    {
        return x.Map(x => Math.Max(0.0f, x));
    }

    ///<inheritdoc/>
    public FTensor ActivationDerivative(FTensor x)
    {
        return x.Map(x => x > 0.0f ? 1.0f : 0.0f);
    }
}
