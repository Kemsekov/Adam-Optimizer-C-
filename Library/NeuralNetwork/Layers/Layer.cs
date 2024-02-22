using System.Collections.Immutable;
using Tensornet;

namespace GradientDescentSharp.NeuralNetwork;
/// <summary>
/// Default layer
/// </summary>
public unsafe class Layer : ILayer
{
    ///<inheritdoc/>
    public FTensor Weights { get; }
    ///<inheritdoc/>
    public FTensor Bias { get; }
    /// <summary>
    /// Output without applied activation function. <br/>
    /// </summary>
    /// <value></value>
    public FTensor RawOutput { get; }
    ///<inheritdoc/>
    public IActivationFunction Activation { get; }

    /// <param name="inputSize">Layer input size</param>
    /// <param name="outputSize">Layer output size</param>
    /// <param name="activation">Activation function. May choose from <see cref="ActivationFunction"/></param>
    public Layer(int inputSize, int outputSize, IActivationFunction activation)
    {
        Weights = Tensor.Zeros<float>(new(outputSize, inputSize));
        Bias = Tensor.Zeros<float>(new(outputSize, 1));
        RawOutput = Tensor.Zeros<float>(new(outputSize, 1));

        Activation = activation;
        Activation.WeightsInit.InitBiasWeights(Bias);
        Activation.WeightsInit.InitWeights(Weights);
    }
    ///<inheritdoc/>
    public FTensor Forward(FTensor input)
    {
        var raw = Weights.Matmul(input) + Bias;
        return raw;
    }
    ///<inheritdoc/>
    public Gradient ComputeGradient(FTensor layerInput, FTensor layerOutput, FTensor inputLossDerivative, bool updateLossDerivative, out FTensor? newLossDerivative)
    {
        var activation = Activation.Activation;
        var derivative = Activation.ActivationDerivative;

        var layerOutputDerivative = derivative(layerOutput);
        var span = inputLossDerivative.AsSpan();
        var biasesGradient = layerOutputDerivative.VecMap(span,(i,s, v) => s[i] * v);

        newLossDerivative = null;
        if (updateLossDerivative)
        {
            var layerOutputDerivativeSpan = layerOutputDerivative.AsSpan();
            inputLossDerivative.VecMapInplace(layerOutputDerivativeSpan,(j,s, x) => x * s[j]);
            newLossDerivative = inputLossDerivative.Matmul(Weights);
        }
        return new(biasesGradient, layerInput);
    }
}
