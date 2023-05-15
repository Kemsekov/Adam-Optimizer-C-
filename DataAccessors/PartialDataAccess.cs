namespace GradientDescentSharp.DataAccessors;

/// <summary>
/// Allows to take some part of full data access, and threat it as another 
/// closed data accessor.
/// </summary>
public class PartialDataAccess<T> : IDataAccess<T>
{
    public PartialDataAccess(IDataAccess<T> data, int startIndex, int length)
    {
        Data = data;
        StartIndex = startIndex;
        Length = length;
    }
    public T this[int index]
    {
        get => Data[StartIndex + index];
        set => Data[StartIndex + index] = value;
    }
    public int Length { get; init; }
    public IDataAccess<T> Data { get; }
    public int StartIndex { get; }

    public IEnumerator<T> GetEnumerator()
    {
        for (int i = 0; i < Length; i++)
            yield return this[i];
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}