using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork;
/// <summary>
/// Wrapper around neural network that limits it's capabilities to only predicting values
/// </summary>
public  class PredictOnlyNN
{
    protected NNBase nnBase;

    public PredictOnlyNN(NNBase nnBase)
    {
        this.nnBase = nnBase;
    }
    public virtual Vector Forward(Vector input)=> nnBase.Forward(input);
    public virtual float Error(Vector input, Vector expected)=>nnBase.Error(input,expected);

}