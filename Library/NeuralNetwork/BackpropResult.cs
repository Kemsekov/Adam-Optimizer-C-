namespace GradientDescentSharp.NeuralNetwork;

/// <summary>
/// Backpropogation result. Can be used to unlearn results of backpropagation.
/// </summary>
public class BackpropResult
{
    private ILayer[] layers;
    public BackpropResult(ILayer[] layers)
    {
        this.layers = layers;
    }
    /// <summary>
    /// Unlearns last backpropagation results.
    /// </summary>
    public void Unlearn()
    {
        foreach (var l in layers)
            l.Unlearn();
        layers = Array.Empty<ILayer>();
    }
}
