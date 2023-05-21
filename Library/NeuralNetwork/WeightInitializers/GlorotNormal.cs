namespace GradientDescentSharp.NeuralNetwork.WeightInitializers;
using MathNet.Numerics.LinearAlgebra.Single;

public class GlorotNormal : IWeightsInit
{
    public Random Rand { get; set;}

    public GlorotNormal(Random? rand = null)
    {
        this.Rand = rand ?? new Random();
    }
    public void InitWeights(Vector bias)
    {
        bias.MapInplace(x=>Rand.NextSingle());
    }
    public void InitWeights(Matrix layer)
    {
        var stddev = Math.Sqrt(2.0/(layer.RowCount+layer.ColumnCount));
        layer.MapInplace(x=> (float)MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev));
    }

    public float SampleWeight(Matrix layer)
    {
        var stddev = Math.Sqrt(2.0/(layer.RowCount+layer.ColumnCount));
        return (float)MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev);
    }
}

