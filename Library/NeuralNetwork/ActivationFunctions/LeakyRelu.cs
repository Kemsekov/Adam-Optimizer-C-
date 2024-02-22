
namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

///<inheritdoc/>
public class LeakyRelu : IActivationFunction
{
    private float _alpha;
    /// <param name="alpha">Coefficient of line at x smaller than 0</param>
    public LeakyRelu(float alpha)
    {
        _alpha=alpha;
    }
    ///<inheritdoc/>
    public IWeightsInit WeightsInit{get;set;}= new He2Normal();
    ///<inheritdoc/>
    public FTensor Activation(FTensor x)
    {
        return x.Map(x=>Math.Max(_alpha*x,x));
    }
    ///<inheritdoc/>
    public FTensor ActivationDerivative(FTensor x)
    {
        return x.Map(x=>x>0 ? 1.0f : _alpha);
    }
}
