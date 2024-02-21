
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
    FVector Activation(FVector x);
    /// <summary>
    /// </summary>
    FVector ActivationDerivative(FVector x);
}
