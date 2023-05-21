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
    public void InitWeights(Vector bias)
    {
        bias.MapInplace(x=>Rand.NextSingle());
    }
    public void InitWeights(Matrix layer)
    {
        var stddev = Math.Sqrt(2.0/layer.ColumnCount);
        var vv = MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev);
        layer.MapInplace(x=> (float)MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev));
    }
}

public class He2Normal : IWeightsInit
{
    public Random Rand { get; set;}

    public He2Normal(Random? rand = null)
    {
        this.Rand = rand ?? new Random();
    }
    public void InitWeights(Vector bias)
    {
        bias.MapInplace(x=>Rand.NextSingle());
    }
    public void InitWeights(Matrix layer)
    {
        var stddev = Math.Sqrt(1.0/layer.ColumnCount);
        var vv = MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev);
        layer.MapInplace(x=> (float)MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev));
    }
}

public class He3Normal : IWeightsInit
{
    public Random Rand { get; set;}

    public He3Normal(Random? rand = null)
    {
        this.Rand = rand ?? new Random();
    }
    public void InitWeights(Vector bias)
    {
        bias.MapInplace(x=>Rand.NextSingle());
    }
    public void InitWeights(Matrix layer)
    {
        var stddev = Math.Sqrt(6.0/layer.ColumnCount);
        var vv = MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev);
        layer.MapInplace(x=> (float)MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev));
    }
}
