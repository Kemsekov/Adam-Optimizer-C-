namespace GradientDescentSharp.NeuralNetwork;

public interface IActivationFunction{
    FloatType Activation(FloatType x);
    FloatType ActivationDerivative(FloatType x);
}
