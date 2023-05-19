namespace GradientDescentSharp.NeuralNetwork;
public interface ILayer
{
    Matrix Weights{get;}
    Vector Bias{get;}
    IActivationFunction Activation{get;}
    Vector Forward(Vector input);
    void Learn(Vector biasesGradient, Vector layerInput, double learningRate);
    public void Unlearn();
}
