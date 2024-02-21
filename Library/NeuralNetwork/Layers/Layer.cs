namespace GradientDescentSharp.NeuralNetwork;
/// <summary>
/// Default layer
/// </summary>
public class Layer : ILayer
{
    ///<inheritdoc/>
    public FMatrix Weights { get; }
    ///<inheritdoc/>
    public FVector Bias { get; }
    /// <summary>
    /// Output without applied activation function. <br/>
    /// </summary>
    /// <value></value>
    public FVector RawOutput { get; }
    ///<inheritdoc/>
    public IActivationFunction Activation { get; }

    /// <param name="factory">Linear objects factory</param>
    /// <param name="inputSize">Layer input size</param>
    /// <param name="outputSize">Layer output size</param>
    /// <param name="activation">Activation function. May choose from <see cref="ActivationFunction"/></param>
    public Layer(IComplexObjectsFactory<float> factory, int inputSize, int outputSize, IActivationFunction activation)
    {
        Weights = factory.CreateMatrix(outputSize, inputSize);
        Bias = factory.CreateVector(outputSize);
        RawOutput = factory.CreateVector(outputSize);
        Activation = activation;
        Activation.WeightsInit.InitWeights(Bias);
        Activation.WeightsInit.InitWeights(Weights);
    }
    ///<inheritdoc/>
    public FVector Forward(FVector input)
    {
        var raw = Weights * input + Bias;
        return raw;
    }
    ///<inheritdoc/>
    public Gradient ComputeGradient(FVector layerInput,FVector layerOutput,FVector inputLossDerivative, bool updateLossDerivative, out FVector? newLossDerivative)
    {
        var activation = Activation.Activation;
        var derivative = Activation.ActivationDerivative;

        var layerOutputDerivative = derivative(layerOutput);
        var biasesGradient = layerOutputDerivative.MapIndexed((j, x) => x * inputLossDerivative[j]);
        newLossDerivative=null;
        if (updateLossDerivative)
        {
            inputLossDerivative.MapIndexedInplace((j, x) => x * layerOutputDerivative[j]);
            newLossDerivative = inputLossDerivative*Weights;
        }
        return new(biasesGradient,layerInput);
    }
}
