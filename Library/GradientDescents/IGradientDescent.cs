namespace GradientDescentSharp.GradientDescents;

public interface IGradientDescent
{
    /// <summary>
    /// Does gradient descent
    /// </summary>
    /// <param name="maxIterations">What is upper limit of steps gradient descent can do</param>
    /// <returns>Number of steps made</returns>
    public int Descent(int maxIterations);
}