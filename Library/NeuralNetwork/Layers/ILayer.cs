using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork;
public interface ILayer
{
    IWeightsInit WeightsInit{get;}
    Matrix Weights{get;}
    Vector Bias{get;}
    IActivationFunction Activation{get;}
    Vector Forward(Vector input);
    void Learn(Vector biasesGradient, Vector layerInput, float learningRate);
    public void Unlearn();
}
