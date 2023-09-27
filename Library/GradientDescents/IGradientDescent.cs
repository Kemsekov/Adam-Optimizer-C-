using System.Numerics;


namespace GradientDescentSharp.GradientDescents;

/// <summary>
/// Gradient descent interface
/// </summary>
public interface IGradientDescent
{
    /// <summary>
    /// Does gradient descent
    /// </summary>
    /// <returns>Enumerable that can be used to track changes in descent.</returns>
    IEnumerable<IDescentStep> Descent();
    /// <summary>
    /// Does gradient descent
    /// </summary>
    /// <param name="maxIterations">What is upper limit of steps gradient descent can do</param>
    /// <returns>Number of steps made</returns>
    public int Descent(int maxIterations);
}