namespace GradientDescentSharp.NeuralNetwork.WeightInitializers;

/// <summary>
/// Weight initialization
/// </summary>
public interface IWeightsInit{
    ///<inheritdoc/>
    public void InitWeights(FVector bias);
    ///<inheritdoc/>
    public void InitWeights(FMatrix layer);
    /// <summary>
    /// Get a single sample that could be assigned to weights in given layer weights
    /// </summary>
    public float SampleWeight(FMatrix layer);
}