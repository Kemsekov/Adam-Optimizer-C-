namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

    ///<inheritdoc/>
public class Linear : IActivationFunction
{
    ///<inheritdoc/>
    public IWeightsInit WeightsInit{get;set;} = new Guassian();
    ///<inheritdoc/>
    public FVector Activation(FVector x)
    {
        return x.Map(x=>x);
    }

    ///<inheritdoc/>
    public FVector ActivationDerivative(FVector x)
    {
        return x.Map(x=>1.0f);
    }
}
