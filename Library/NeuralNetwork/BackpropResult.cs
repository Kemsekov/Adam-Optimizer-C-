namespace GradientDescentSharp.NeuralNetwork;

/// <summary>
/// Backpropogation result. Can be used to unlearn results of backpropagation.
/// </summary>
public class BackpropResult
{
    private IEnumerable<Learned> learned;
    public BackpropResult(IEnumerable<Learned> learned)
    {
        this.learned = learned;
    }
    /// <summary>
    /// Unlearns last backpropagation results.
    /// </summary>
    public void Unlearn()
    {
        foreach (var l in learned)
            l.Unlearn();
        learned = Array.Empty<Learned>();
    }
}
