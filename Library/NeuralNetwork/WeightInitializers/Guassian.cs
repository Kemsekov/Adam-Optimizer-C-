namespace GradientDescentSharp.NeuralNetwork.WeightInitializers;

/// <summary>
/// Uses random values drawn from a Gaussian distribution with mean 0 and standard deviation 1 (also known as the standard normal distribution).
/// </summary>
public class Guassian: IWeightsInit
{
    ///<inheritdoc/>
    public Random Rand { get; set;}

    ///<inheritdoc/>
    public Guassian(Random? rand = null)
    {
        this.Rand = rand ?? new Random();
    }
    ///<inheritdoc/>
    public void InitBiasWeights(FTensor bias)
    {
        bias.MapInplace(x=>Rand.NextSingle());
    }
    ///<inheritdoc/>
    public void InitWeights(FTensor layer)
    {
        var stddev = Math.Sqrt(2.0/(layer.Shape[0]+layer.Shape[1]));
        layer.MapInplace(x=> (float)MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev));
    }

    ///<inheritdoc/>
    public float SampleWeight(FTensor layer)
    {
        var stddev = Math.Sqrt(2.0/(layer.Shape[0]+layer.Shape[1]));
        return (float)MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev);
    }
}