namespace GradientDescentSharp.GradientDescents;

/// <summary>
/// Descent step information
/// </summary>
public interface IDescentStep
{
    /// <summary>
    /// Loss function value at current descent step
    /// </summary>
    public double Loss { get; }
    /// <summary>
    /// How much loss function changed from this step
    /// </summary>
    public double Difference { get; }
    /// <summary>
    /// Current step iteration
    /// </summary>
    public long Iteration { get; }
    /// <summary>
    /// true if given iteration have overshoot previous loss function minima.<br/>
    /// In most cases descents here implement auto-rollback for this scenarios, so you 
    /// don't need to handle this much.
    /// </summary>
    public bool Overshoot { get; }
}
///<inheritdoc cref="IDescentStep"/>
/// <param name="Loss">Loss function value at current descent step</param>
/// <param name="Difference">How much loss function changed from this step</param>
/// <param name="Iteration">Current step iteration</param>
/// <param name="Overshoot">
/// true if given iteration have overshoot previous loss function minima.<br/>
/// In most cases descents here implement auto-rollback for this scenarios, so you 
/// don't need to handle this much.
/// </param>
public record DescentStep(double Loss, double Difference, long Iteration, bool Overshoot) : IDescentStep;
