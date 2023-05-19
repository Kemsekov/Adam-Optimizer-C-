namespace GradientDescentSharp.NeuralNetwork.WeightInitializers;

public class GlorotUniform : IWeightsInit
{
    public Random Rand { get; set;}

    public GlorotUniform(Random? rand = null)
    {
        this.Rand = rand ?? new Random();
    }
    public void InitWeights(Vector bias)
    {
        bias.MapInplace(x=>Rand.NextDouble());
    }
    public void InitWeights(Matrix layer)
    {
        var limit = Math.Sqrt(6.0/(layer.RowCount+layer.ColumnCount));
        layer.MapInplace(x=>(Rand.NextDouble()*2-1)*limit);
    }
}

