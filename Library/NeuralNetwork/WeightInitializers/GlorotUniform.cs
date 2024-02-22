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
    public void InitBiasWeights(FTensor bias)
    {
        bias.MapInplace(x=>Rand.NextSingle());
    }
    ///<inheritdoc/>
    public void InitWeights(FTensor layer)
    {
        var limit = Math.Sqrt(6.0/(layer.Shape[0]+layer.Shape[1]));
        layer.MapInplace(x=> (float)((Rand.NextSingle()*2-1)*limit));
    }

    ///<inheritdoc/>
    public float SampleWeight(FTensor layer)
    {
        var limit = Math.Sqrt(6.0/(layer.Shape[0]+layer.Shape[1]));
        return (float)((Rand.NextSingle()*2-1)*limit);
    }
}

