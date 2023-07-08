
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork;

public class Layer : ILayer
{
    public FMatrix Weights { get; }
    public FVector Bias { get; }
    /// <summary>
    /// Output without applied activation function. <br/>
    /// To get true output call
    /// <see cref="ActivatedOutput"/>
    /// </summary>
    /// <value></value>
    public FVector RawOutput { get; }
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
        Bias = factory.CreateVector(outputSize);
        RawOutput = factory.CreateVector(outputSize);
        Activation = activation;
        weightsInit.InitWeights(Bias);
        weightsInit.InitWeights(Weights);
        WeightsInit = weightsInit;
    }
    public FVector Forward(FVector input)
    {
        var raw = Weights * input + Bias;
        return raw;
    }

    public Gradient ComputeGradient(FVector layerOutput,FVector inputLossDerivative)
    {
        var activation = Activation.Activation;
        var derivative = Activation.ActivationDerivative;

        var layerOutputDerivative = derivative(layerOutput);
        var biasesGradient = layerOutputDerivative.MapIndexed((j, x) => x * inputLossDerivative[j]);
        throw new NotImplementedException();
        // if (i > 0)
        // {
        //     inputLossDerivative.MapIndexedInplace((j, x) => x * layerOutputDerivative[j]);
        //     Weights.LeftMultiply(inputLossDerivative,inputLossDerivative);
        // }
        // var layerInput = i > 0 ? activation(rawLayersOutput[Layers[i - 1]]) : input.Clone();

    }
}
