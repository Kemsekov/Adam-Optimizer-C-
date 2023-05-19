namespace GradientDescentSharp.NeuralNetwork;

public interface IActivationFunction{
    double Activation(double x);
    double ActivationDerivative(double x);
}
