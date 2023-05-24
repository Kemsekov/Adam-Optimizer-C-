
using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork;

public class Layer : ILayer
{
    public Matrix Weights { get; }
    public Vector Bias { get; }
    /// <summary>
    /// Output without applied activation function. <br/>
    /// To get true output call
    /// <see cref="ActivatedOutput"/>
    /// </summary>
    /// <value></value>
    public Vector RawOutput { get; }
    public IActivationFunction Activation { get; }

    public IWeightsInit WeightsInit { get; }

    /// <param name="factory">Linear objects factory</param>
    /// <param name="inputSize">Layer input size</param>
    /// <param name="outputSize">Layer output size</param>
    /// <param name="activation">Activation function. May choose from <see cref="ActivationFunction"/></param>
    /// <param name="weightsInit">Weight initialization.</param>
    public Layer(IComplexObjectsFactory<float> factory, int inputSize, int outputSize, IActivationFunction activation, IWeightsInit weightsInit)
    {
        Weights = (Matrix)factory.CreateMatrix(outputSize, inputSize);
        Bias = (Vector)factory.CreateVector(outputSize);
        RawOutput = (Vector)factory.CreateVector(outputSize);
        Activation = activation;
        weightsInit.InitWeights(Bias);
        weightsInit.InitWeights(Weights);
        WeightsInit = weightsInit;
    }
    public Vector Forward(Vector input)
    {
        var raw = Weights * input + Bias;
        return (Vector)raw;
    }

}
