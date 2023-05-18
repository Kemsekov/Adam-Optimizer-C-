namespace GradientDescentSharp.NeuralNetwork;

public class BackpropResult
{
    private ILayer[] layers;
    public BackpropResult(ILayer[] layers)
    {
        this.layers = layers;
    }
    public void Unlearn()
    {
        foreach (var l in layers)
            l.Unlearn();
    }
}
