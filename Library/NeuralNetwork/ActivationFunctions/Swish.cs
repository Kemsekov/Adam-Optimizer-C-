namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

///<inheritdoc/>
public class Swish : IActivationFunction
{
    ///<inheritdoc/>
    public IWeightsInit WeightsInit { get; set; } = new He3Normal();
    private float beta;

    ///<inheritdoc/>
    public Swish(float beta)
    {
        this.beta = beta;
    }
    ///<inheritdoc/>
    public FTensor Activation(FTensor x)
    {
        return x.Map(x => x / (1 + MathF.Exp(-beta * x)));
    }

    ///<inheritdoc/>
    public FTensor ActivationDerivative(FTensor x)
    {
        return x.Map(x =>
        {
            x *= beta;
            var sigmoid = 1.0f / (1 + MathF.Exp(-x));
            var sigmoidDerivative = 1 - sigmoid;
            return sigmoid + x * sigmoidDerivative;
        });
    }
}
