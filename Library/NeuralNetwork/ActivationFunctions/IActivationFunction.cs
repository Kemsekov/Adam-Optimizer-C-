
using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork;

public interface IActivationFunction{
    Vector Activation(Vector x);
    Vector ActivationDerivative(Vector x);
}
