using System.Numerics;
namespace GradientDescentSharp.Utils;

/// <summary>
/// Math generic operations
/// </summary>
/// <typeparam name="TFloat"></typeparam>
public static class Math<TFloat>
    where TFloat : INumber<TFloat>
{
    /// <summary>
    /// Raises TFloat value to a whole power
    /// </summary>
    public static TFloat Pow(TFloat value, int pow)
    {
        var result = TFloat.One;
        for(int i = 0;i<pow;i++)
            result *=value;
        return result;
    }
    /// <summary>
    /// Try to cast value to double, computes sqrt by <see cref="Math.Sqrt"/> and tries to cast double back to TFloat.
    /// It will break only at runtime because I use here dynamic
    /// </summary>
    public static TFloat Sqrt(TFloat value){
        return (TFloat)(Math.Sqrt((double)(value as dynamic)) as dynamic);
    }
    /// <returns>Absolute value</returns>
    public static TFloat Abs(TFloat value){
        return value>=TFloat.Zero ? value : -value;
    }
}