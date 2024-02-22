namespace GradientDescentSharp.NeuralNetwork.WeightInitializers;

/// <summary>
/// Weight initialization
/// </summary>
public interface IWeightsInit{
    ///<inheritdoc/>
    public void InitWeights(FTensor layer);
    ///<inheritdoc/>
    public void InitBiasWeights(FTensor layer);
    /// <summary>
    /// Get a single sample that could be assigned to weights in given layer weights
    /// </summary>
    public float SampleWeight(FTensor layer);
}