namespace GradientDescentSharp.DataAccessors;

/// <summary>
/// When computing gradients we need to change some value of variables by epsilon <br/>
/// and recompute function on it.  <br/>
/// Because re-creating a new data variables with some small change of single value is
/// not efficient, this class allows to create a new IDataAccess that delegates this logic.<br/>
/// It does only one thing: on some index instead of original value it
/// returns changed value. <br/>
/// By doing this we avoid all problems with memory and it also allows us to
/// compute gradients in parallel!
/// </summary>
public class GradientDataAccess<T> : IDataAccess<T>
{
    /// <param name="original">Original data</param>
    /// <param name="changedIndex">Index of a variable we need to replace</param>
    /// <param name="changedValue">Replaced variable</param>
    public GradientDataAccess(IDataAccess<T> original, int changedIndex, T changedValue)
    {
        DataAccess = original;
        ChangedIndex = changedIndex;
        ChangedValue = changedValue;
    }

    public T this[int index]
    {
        get => (index == ChangedIndex) ? ChangedValue : DataAccess[index];
        //I decided to allow set operation by overriding changed value too.
        //so far I've not seen major troubles with such approach
        set
        {
            DataAccess[index] = value;
            if(index == ChangedIndex)
                ChangedValue = value;
        }
    }
    /// <summary>
    /// Resets replaced value with new one
    /// </summary>
    /// <param name="changedIndex">Index of a variable we need to replace</param>
    /// <param name="changedValue">Replaced variable</param>
    public void Reset(int changedIndex, T changedValue)
    {
        ChangedIndex = changedIndex;
        ChangedValue = changedValue;
    }
    public IDataAccess<T> DataAccess { get; }
    public int ChangedIndex { get; protected set; }
    public T ChangedValue { get; protected set; }
    public int Length => DataAccess.Length;

    public IEnumerator<T> GetEnumerator()
    {
        for (int i = 0; i < Length; i++)
        {
            yield return this[i];
        }
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}