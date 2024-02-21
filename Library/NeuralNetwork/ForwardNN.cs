namespace GradientDescentSharp.NeuralNetwork;
/// <summary>
/// Feed-forward neural network
/// </summary>
public class ForwardNN : NNBase
{
    ///<inheritdoc/>
    public ForwardNN(params ILayer[] layers) : base(layers)
    {
    }
}