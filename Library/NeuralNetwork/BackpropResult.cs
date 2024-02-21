namespace GradientDescentSharp.NeuralNetwork;

/// <summary>
/// Backpropogation result. Can be used to learn and unlearn results of backpropagation.<br/>
/// Call <see cref="Learn"/> to learn last backpropagation computation<br/>
/// Call <see cref="Unlearn"/> to unlearn last backpropagation computation<br/>
/// You can use these methods in any order with different instances of <see cref="BackpropResult"/>, as much as you want, so be sure to count
/// learn and unlearn methods count even.
/// </summary>
public class BackpropResult
{
    private IEnumerable<ILearner> learner;
    ///<inheritdoc/>
    public BackpropResult(IEnumerable<ILearner> learner)
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
