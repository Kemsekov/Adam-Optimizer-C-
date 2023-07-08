namespace GradientDescentSharp.NeuralNetwork.WeightInitializers;

public interface IWeightsInit{
    public void InitWeights(FVector bias);
    public void InitWeights(FMatrix layer);
    /// <summary>
    /// Get a single sample that could be assigned to weights in given layer weights
    /// </summary>
    public float SampleWeight(FMatrix layer);
}