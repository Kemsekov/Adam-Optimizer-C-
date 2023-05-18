namespace GradientDescentSharp.NeuralNetwork;

public class HeNormal : IWeightsInit
{
    public Random Rand { get; set;}

    public HeNormal(Random? rand = null)
    {
        this.Rand = rand ?? new Random();
    }
    public void InitWeights(Vector bias)
    {
        bias.MapInplace(x=>Rand.NextDouble());
    }
    public void InitWeights(Matrix layer)
    {
        var stddev = Math.Sqrt(2.0/layer.ColumnCount);
        var vv = MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev);
        layer.MapInplace(x=>MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev));
    }
}

