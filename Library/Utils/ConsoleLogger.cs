namespace GradientDescentSharp.Utils;
/// <summary>
/// Simple console logger
/// </summary>
public class ConsoleLogger : ILogger
{
    ///<inheritdoc/>
    public void Log(string message)
    {
        Console.Write(message);
    }
    ///<inheritdoc/>
    public void LogLine(string message)
    {
        Console.WriteLine(message);
    }
}
