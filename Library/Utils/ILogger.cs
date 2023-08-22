namespace GradientDescentSharp.Utils;
/// <summary>
/// Simple logger interface
/// </summary>
public interface ILogger
{
    /// <summary>
    /// Log message without newline
    /// </summary>
    public void Log(string message);
    /// <summary>
    /// Log message with newline
    /// </summary>
    public void LogLine(string message);
}
