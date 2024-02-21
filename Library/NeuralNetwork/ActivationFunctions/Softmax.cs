namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

///<inheritdoc/>
public class Softmax : IActivationFunction
{
    ///<inheritdoc/>
    public IWeightsInit WeightsInit { get; set; } = new HeNormal();
    ///<inheritdoc/>
    public FVector Activation(FVector x)
    {
        var sum = x.Sum(x => MathF.Exp(x));
        return x.Map(x => MathF.Exp(x) / sum);
    }

    ///<inheritdoc/>
    public FVector ActivationDerivative(FVector x)
    {
        var sum = x.Sum(x => MathF.Exp(x));
        return x.Map(x =>
        {
            var ex = MathF.Exp(x);
            var variable = ex / sum;
            return variable - variable * variable;
        });
    }
}
