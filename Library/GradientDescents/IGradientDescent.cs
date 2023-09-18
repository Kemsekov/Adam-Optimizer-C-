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
    /// <returns>Enumerable that can be used to do iterations. So to do 10 descent steps just write Descent(100).Take(10)</returns>
    IEnumerable<int> Descent();
    /// <summary>
    /// Does gradient descent
    /// </summary>
    /// <param name="maxIterations">What is upper limit of steps gradient descent can do</param>
    /// <returns>Number of steps made</returns>
    public int Descent(int maxIterations);
}