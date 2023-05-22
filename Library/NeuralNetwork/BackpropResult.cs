namespace GradientDescentSharp.NeuralNetwork;

/// <summary>
/// Backpropogation result. Can be used to unlearn results of backpropagation.
/// </summary>
public class BackpropResult
{
    private IEnumerable<Learner> learner;
    public BackpropResult(IEnumerable<Learner> learner)
    {
        this.learner = learner;
    }
    /// <summary>
    /// Learns backpropagation results.
    /// </summary>
    public void Learn(){
        foreach (var l in learner)
            l.Learn();
    }
    /// <summary>
    /// Unlearns backpropagation results.
    /// </summary>
    public void Unlearn()
    {
        foreach (var l in learner)
            l.Unlearn();
    }
}
