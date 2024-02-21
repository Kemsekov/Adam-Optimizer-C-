namespace GradientDescentSharp.DataAccessors;
/// <summary>
/// Data accessor that wraps around array
/// </summary>
/// <typeparam name="T"></typeparam>
public class ArrayDataAccess<T> : IDataAccess<T>
{
    /// <summary>
    /// Creates new <see cref="ArrayDataAccess{T}"/> 
    /// </summary>
    public ArrayDataAccess(T[] array)
    {
        Array = array;
    }
    /// <summary>
    /// Will create an array of given length
    /// </summary>
    public ArrayDataAccess(int length)
    {
        Array = new T[length];
    }
    /// <summary>
    /// Get and set value under given index
    /// </summary>
    public T this[int index]
    {
        get => Array[index];
        set => Array[index] = value;
    }
    /// <summary>
    /// Base array
    /// </summary>
    public T[] Array { get; }
    /// <summary>
    /// Length of base array
    /// </summary>
    public int Length => Array.Length;
    ///<inheritdoc/>
    public IEnumerator<T> GetEnumerator()
    {
        foreach (var v in Array) yield return v;
    }
    IEnumerator IEnumerable.GetEnumerator() => Array.GetEnumerator();
    ///<inheritdoc/>
    public static implicit operator T[](ArrayDataAccess<T> t) => t.Array;
    ///<inheritdoc/>
    public static implicit operator ArrayDataAccess<T>(T[] t) => new(t);
    ///<inheritdoc/>
    public override string ToString()
    {
        return "ArrayDataAccess<"+typeof(T)+">\n"+string.Join("\n",Array);
    }
}