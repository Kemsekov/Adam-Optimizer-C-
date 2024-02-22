
namespace GradientDescentSharp.NeuralNetwork;

/// <summary>
/// Activation function interface
/// </summary>
public interface IActivationFunction{
    /// <summary>
    /// Weight initialization that is suitable with current activation function
    /// </summary>
    IWeightsInit WeightsInit{get;}
    /// <summary>
    /// </summary>
    FTensor Activation(FTensor x);
    /// <summary>
    /// </summary>
    FTensor ActivationDerivative(FTensor x);
}
