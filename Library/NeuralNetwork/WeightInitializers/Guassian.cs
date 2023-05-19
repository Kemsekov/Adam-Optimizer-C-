namespace GradientDescentSharp.NeuralNetwork.WeightInitializers;
/// <summary>
/// Uses random values drawn from a Gaussian distribution with mean 0 and standard deviation 1 (also known as the standard normal distribution).
/// </summary>
public class Guassian: IWeightsInit
{
    public Random Rand { get; set;}

    public Guassian(Random? rand = null)
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
        layer.MapInplace(x=>(MathNet.Numerics.Distributions.Normal.Sample(Rand,0,stddev)));
    }
}