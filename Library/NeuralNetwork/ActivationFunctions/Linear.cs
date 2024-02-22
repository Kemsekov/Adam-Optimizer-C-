namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

    ///<inheritdoc/>
public class Linear : IActivationFunction
{
    ///<inheritdoc/>
    public IWeightsInit WeightsInit{get;set;} = new Guassian();
    ///<inheritdoc/>
    public FTensor Activation(FTensor x)
    {
        return x.Map(x=>x);
    }

    ///<inheritdoc/>
    public FTensor ActivationDerivative(FTensor x)
    {
        return x.Map(x=>1.0f);
    }
}
