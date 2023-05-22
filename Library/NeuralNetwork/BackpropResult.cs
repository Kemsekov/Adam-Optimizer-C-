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
    public void Learn(){
        foreach (var l in learner)
            l.Learn();
    }
    /// <summary>
    /// Unlearns last backpropagation results.
    /// </summary>
    public void Unlearn()
    {
        foreach (var l in learner)
            l.Unlearn();
        learner = Array.Empty<Learner>();
    }
}
