
using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork;

public record Learner(ILayer layer, Vector biasesGradient, Vector layerInput, float learningRate)
{
    bool unlearned = false;
    bool learned = false;
    public void Learn()
    {
        if(learned) return;
        learned = true;
        for (int k = 0; k < layerInput.Count; k++)
        {
            var kInput = layerInput[k];
            if (kInput == 0) continue;
            for (int j = 0; j < layer.Weights.RowCount; j++)
            {
                var weightGradient = biasesGradient[j] * kInput;
                layer.Weights[j, k] -= learningRate * weightGradient;
            }
        }
        layer.Bias.MapIndexedInplace((j, x) => x - learningRate * biasesGradient[j]);
    }
    /// <summary>
    /// Unlearns last learned weights and biases
    /// </summary>
    public void Unlearn()
    {
        if (unlearned) return;
        unlearned = true;
        var weightsGradient = (int j, int k) => biasesGradient[j] * layerInput[k];
        layer.Weights.MapIndexedInplace((j, k, x) => x + learningRate * weightsGradient(j, k));
        layer.Bias.MapIndexedInplace((j, x) => x + learningRate * biasesGradient[j]);
    }
}

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
