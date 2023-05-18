namespace GradientDescentSharp.NeuralNetwork;

public class GlorotNormal : IWeightsInit
{
    public Random Rand { get; set;}

    public GlorotNormal(Random? rand = null)
    {
        this.Rand = rand ?? new Random();
    }
    public void InitWeights(Vector bias)
    {
        bias.MapInplace(x=>Rand.NextDouble());
    }
    public void InitWeights(Matrix layer)
    {
        var stddev = Math.Sqrt(2.0/(layer.RowCount+layer.ColumnCount));
        layer.MapInplace(x=>MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev));
    }
}

