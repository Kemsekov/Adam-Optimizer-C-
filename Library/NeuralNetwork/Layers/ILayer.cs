using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork;
public interface ILayer
{
    IWeightsInit WeightsInit{get;}
    FMatrix Weights{get;}
    FVector Bias{get;}
    IActivationFunction Activation{get;}
    FVector Forward(FVector input);
    /// <summary>
    /// Computes gradients for current layer
    /// </summary>
    /// <param name="layerOutput">Current layer output for previous NN feed</param>
    /// <param name="inputLossDerivative">Current layer input loss function derivative</param>
    /// <returns>Gradients for current layer</returns>
    Gradient ComputeGradient(FVector layerOutput,FVector inputLossDerivative);
}
