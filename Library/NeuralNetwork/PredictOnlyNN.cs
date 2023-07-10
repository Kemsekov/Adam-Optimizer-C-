namespace GradientDescentSharp.NeuralNetwork;
/// <summary>
/// Wrapper around neural network that limits it's capabilities to only predicting values
/// </summary>
public class PredictOnlyNN
{
    ///<inheritdoc/>
    protected NNBase nnBase;
    /// <summary>
    /// Creates a wrapper around neural network to limit it's capabilities to only prediction
    /// </summary>
    /// <param name="nnBase"></param>
    public PredictOnlyNN(NNBase nnBase)
    {
        this.nnBase = nnBase;
    }
    ///<inheritdoc/>
    public FVector Forward(FVector input)=> nnBase.Forward(input);
    ///<inheritdoc/>
    public float Error(FVector input, FVector expected)=>nnBase.Error(input,expected);

}