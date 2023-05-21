namespace GradientDescentSharp.NeuralNetwork;

public interface IActivationFunction{
    float Activation(float x);
    float ActivationDerivative(float x);
}
