using MathNet.Numerics.LinearAlgebra.Single;
using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork.WeightInitializers;

public class HeNormal : IWeightsInit
{
    public Random Rand { get; set;}

    public HeNormal(Random? rand = null)
    {
        this.Rand = rand ?? new Random();
    }
    public void InitWeights(FVector bias)
    {
        bias.MapInplace(x=>Rand.NextSingle());
    }
    public void InitWeights(FMatrix layer)
    {
        var stddev = Math.Sqrt(2.0/layer.ColumnCount);
        layer.MapInplace(x=> (float)MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev));
    }

    public float SampleWeight(FMatrix layer)
    {
        var stddev = Math.Sqrt(2.0/layer.ColumnCount);
        return (float)MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev);
    }
}

public class He2Normal : IWeightsInit
{
    public Random Rand { get; set;}

    public He2Normal(Random? rand = null)
    {
        this.Rand = rand ?? new Random();
    }
    public void InitWeights(FVector bias)
    {
        bias.MapInplace(x=>Rand.NextSingle());
    }
    public void InitWeights(FMatrix layer)
    {
        var stddev = Math.Sqrt(1.0/layer.ColumnCount);
        layer.MapInplace(x=> (float)MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev));
    }

    public float SampleWeight(FMatrix layer)
    {
        var stddev = Math.Sqrt(1.0/layer.ColumnCount);
        return (float)MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev);
    }
}

public class He3Normal : IWeightsInit
{
    public Random Rand { get; set;}

    public He3Normal(Random? rand = null)
    {
        this.Rand = rand ?? new Random();
    }
    public void InitWeights(FVector bias)
    {
        bias.MapInplace(x=>Rand.NextSingle());
    }
    public void InitWeights(FMatrix layer)
    {
        var stddev = Math.Sqrt(6.0/layer.ColumnCount);
        layer.MapInplace(x=> (float)MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev));
    }

    public float SampleWeight(FMatrix layer)
    {
        var stddev = Math.Sqrt(6.0/layer.ColumnCount);
        return (float)MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev);
    }
}
