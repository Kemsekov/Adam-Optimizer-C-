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
    ///<inheritdoc/>
    public IActivationFunction Activation { get; }

    /// <param name="inputSize">Layer input size</param>
    /// <param name="outputSize">Layer output size</param>
    /// <param name="activation">Activation function. May choose from <see cref="ActivationFunction"/></param>
    public Layer(int inputSize, int outputSize, IActivationFunction activation)
    {
        Weights = Tensor.Zeros<float>(new(outputSize, inputSize));
        Bias = Tensor.Zeros<float>(new(outputSize, 1));

        Activation = activation;
        Activation.WeightsInit.InitBiasWeights(Bias);
        Activation.WeightsInit.InitWeights(Weights);
    }
    ///<inheritdoc/>
    public FTensor Forward(FTensor input)
    {
        //result = Weights @ input + Bias
        var result = Weights.Matmul(input);
        result.VecMapInplace(Bias.AsSpan(),(j,s,v)=>v+s[j]);
        return result;
    }
    ///<inheritdoc/>
    public Gradient ComputeGradient(FTensor layerInput, FTensor layerOutput, FTensor inputLossDerivative, bool updateLossDerivative, out FTensor? newLossDerivative)
    {
        var derivative = Activation.ActivationDerivative;

        var layerOutputDerivative = derivative(layerOutput);

        newLossDerivative = null;
        if (updateLossDerivative)
        {
            inputLossDerivative.VecMapInplace(
                layerOutputDerivative.AsSpan(),
                (j,s, x) => x * s[j]
            );

            newLossDerivative = inputLossDerivative.Matmul(Weights);
        }
        
        layerOutputDerivative.VecMapInplace(inputLossDerivative.AsSpan(),(i,s, v) => s[i] * v);
        var biasesGradient = layerOutputDerivative;

        return new(biasesGradient, layerInput);
    }
}
