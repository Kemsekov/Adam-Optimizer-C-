namespace GradientDescentSharp.NeuralNetwork.WeightInitializers;

/// <summary>
/// He normal initialization
/// </summary>
public class HeNormal : IWeightsInit
{
    /// <summary>
    /// What random to use for initialization
    /// </summary>
    public Random Rand { get; set;}
    ///<inheritdoc/>

    public HeNormal(Random? rand = null)
    {
        this.Rand = rand ?? new Random();
    }
    ///<inheritdoc/>
    public void InitWeights(FVector bias)
    {
        bias.MapInplace(x=>Rand.NextSingle());
    }
    ///<inheritdoc/>
    public void InitWeights(FMatrix layer)
    {
        var stddev = Math.Sqrt(2.0/layer.ColumnCount);
        layer.MapInplace(x=> (float)MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev));
    }

    ///<inheritdoc/>
    public float SampleWeight(FMatrix layer)
    {
        var stddev = Math.Sqrt(2.0/layer.ColumnCount);
        return (float)MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev);
    }
}
/// <summary>
/// He 2 normal initialization
/// </summary>
public class He2Normal : IWeightsInit
{
    ///<inheritdoc/>
    public Random Rand { get; set;}

    ///<inheritdoc/>
    public He2Normal(Random? rand = null)
    {
        this.Rand = rand ?? new Random();
    }
    ///<inheritdoc/>
    public void InitWeights(FVector bias)
    {
        bias.MapInplace(x=>Rand.NextSingle());
    }
    ///<inheritdoc/>
    public void InitWeights(FMatrix layer)
    {
        var stddev = Math.Sqrt(1.0/layer.ColumnCount);
        layer.MapInplace(x=> (float)MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev));
    }

    ///<inheritdoc/>
    public float SampleWeight(FMatrix layer)
    {
        var stddev = Math.Sqrt(1.0/layer.ColumnCount);
        return (float)MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev);
    }
}
/// <summary>
/// He 3 normal initialization
/// </summary>
public class He3Normal : IWeightsInit
{
    ///<inheritdoc/>
    public Random Rand { get; set;}

    ///<inheritdoc/>
    public He3Normal(Random? rand = null)
    {
        this.Rand = rand ?? new Random();
    }
    ///<inheritdoc/>
    public void InitWeights(FVector bias)
    {
        bias.MapInplace(x=>Rand.NextSingle());
    }
    ///<inheritdoc/>
    public void InitWeights(FMatrix layer)
    {
        var stddev = Math.Sqrt(6.0/layer.ColumnCount);
        layer.MapInplace(x=> (float)MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev));
    }

    ///<inheritdoc/>
    public float SampleWeight(FMatrix layer)
    {
        var stddev = Math.Sqrt(6.0/layer.ColumnCount);
        return (float)MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev);
    }
}
