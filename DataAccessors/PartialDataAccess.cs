namespace GradientDescentSharp.DataAccessors;

public class PartialDataAccess<T> : IDataAccess<T>
{
    public PartialDataAccess(IDataAccess<T> data, int startIndex, int length)
    {
        this.Data = data;
        this.StartIndex = startIndex;
        this.Length = length;
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