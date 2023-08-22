namespace GradientDescentSharp.DataAccessors;

/// <summary>
/// Allows to take some part of full data access, and threat it as another 
/// closed data accessor.
/// </summary>
public class PartialDataAccess<T> : IDataAccess<T>
{
    /// <summary>
    /// Creates new instance of <see cref="PartialDataAccess{T}"/>
    /// </summary>
    /// <param name="data">Data source</param>
    /// <param name="startIndex">Start index</param>
    /// <param name="length">Length we need to slice</param>
    public PartialDataAccess(IDataAccess<T> data, int startIndex, int length)
    {
        Data = data;
        StartIndex = startIndex;
        Length = length;
    }
    ///<inheritdoc/>
    public T this[int index]
    {
        get => Data[StartIndex + index];
        set => Data[StartIndex + index] = value;
    }
    ///<inheritdoc/>
    public int Length { get; init; }
    /// <summary>
    /// Data source
    /// </summary>
    public IDataAccess<T> Data { get; }
    /// <summary>
    /// Start index of proxy value
    /// </summary>
    public int StartIndex { get; }
    ///<inheritdoc/>
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