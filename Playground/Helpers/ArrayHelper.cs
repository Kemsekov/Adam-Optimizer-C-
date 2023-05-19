namespace Playground;

public static class ArrayHelper
{
    public static IEnumerable<T> TakeNRandom<T>(this T[] values, int n) =>
        Enumerable.Range(0, n).Select(x => values[Random.Shared.Next(values.Length)]);

}