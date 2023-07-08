namespace GradientDescentSharp.NeuralNetwork.WeightInitializers;
using MathNet.Numerics.LinearAlgebra.Single;

public class GlorotUniform : IWeightsInit
{
    public Random Rand { get; set;}

    public GlorotUniform(Random? rand = null)
    {
        this.Rand = rand ?? new Random();
    }
    public void InitWeights(FVector bias)
    {
        bias.MapInplace(x=>Rand.NextSingle());
    }
    public void InitWeights(FMatrix layer)
    {
        var limit = Math.Sqrt(6.0/(layer.RowCount+layer.ColumnCount));
        layer.MapInplace(x=> (float)((Rand.NextSingle()*2-1)*limit));
    }

    public float SampleWeight(FMatrix layer)
    {
        var limit = Math.Sqrt(6.0/(layer.RowCount+layer.ColumnCount));
        return (float)((Rand.NextSingle()*2-1)*limit);
    }
}

