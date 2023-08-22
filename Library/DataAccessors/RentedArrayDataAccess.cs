namespace GradientDescentSharp.DataAccessors;
/// <summary>
/// Data accessor that wraps rented array implementation
/// </summary>
public class RentedArrayDataAccess<T> : IDataAccess<T>, IDisposable
where T : unmanaged
{
    /// <summary>
    /// Creates rented array data access.
    /// </summary>
    public RentedArrayDataAccess(RentedArray<T> array)
    {
        RentedArray = array;
    }
    ///<inheritdoc/>
    public T this[int index]
    {
        get => RentedArray[index];
        set => RentedArray[index] = value;
    }

    ///<inheritdoc/>
    public int Length => RentedArray.Length;
    /// <summary>
    /// Data source that is used
    /// </summary>
    public RentedArray<T> RentedArray { get; }

    ///<inheritdoc/>
    public void Dispose()
    {
        ((IDisposable)RentedArray).Dispose();
    }

    ///<inheritdoc/>
    public IEnumerator<T> GetEnumerator()
    {
        for (int i = 0; i < Length; i++)
        {
            yield return RentedArray[i];
        }
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}
