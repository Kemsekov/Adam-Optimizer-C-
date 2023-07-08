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
    /// <param name="layerInput">Current layer input for previous NN feed</param>
    /// <param name="layerOutput">Current layer output for previous NN feed</param>
    /// <param name="inputLossDerivative">Current layer input loss function derivative</param>
    /// <param name="updateLossDerivative">Used to indicate whether we need to update loss derivative or not</param>
    /// <param name="newLossDerivative">Updated loss derivative, or null if flag for update was set false</param>
    /// <returns>Gradients for current layer</returns>
    Gradient ComputeGradient(FVector layerInput,FVector layerOutput,FVector inputLossDerivative, bool updateLossDerivative, out FVector? newLossDerivative);
}
