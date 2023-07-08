
namespace GradientDescentSharp.NeuralNetwork;

public interface IActivationFunction{
    FVector Activation(FVector x);
    FVector ActivationDerivative(FVector x);
}
