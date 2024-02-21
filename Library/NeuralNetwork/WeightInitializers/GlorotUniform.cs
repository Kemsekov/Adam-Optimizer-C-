namespace GradientDescentSharp.NeuralNetwork.WeightInitializers;
    ///<inheritdoc/>
public class GlorotUniform : IWeightsInit
{
    ///<inheritdoc/>
    public Random Rand { get; set;}

    ///<inheritdoc/>
    public GlorotUniform(Random? rand = null)
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
        var limit = Math.Sqrt(6.0/(layer.RowCount+layer.ColumnCount));
        layer.MapInplace(x=> (float)((Rand.NextSingle()*2-1)*limit));
    }

    ///<inheritdoc/>
    public float SampleWeight(FMatrix layer)
    {
        var limit = Math.Sqrt(6.0/(layer.RowCount+layer.ColumnCount));
        return (float)((Rand.NextSingle()*2-1)*limit);
    }
}

