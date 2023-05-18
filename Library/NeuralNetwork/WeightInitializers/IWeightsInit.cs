namespace GradientDescentSharp.NeuralNetwork;

public interface IWeightsInit{
    public void InitWeights(Vector bias);
    public void InitWeights(Matrix layer);
}