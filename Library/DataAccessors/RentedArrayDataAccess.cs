namespace GradientDescentSharp.DataAccessors;
/// <summary>
/// Data accessor that wraps rented array implementation
/// </summary>
public class RentedArrayDataAccess<T> : IDataAccess<T>, IDisposable
where T : unmanaged
{
    public RentedArrayDataAccess(RentedArray<T> array)
    {
        RentedArray = array;
    }
    public T this[int index]
    {
        get => RentedArray[index];
        set => RentedArray[index] = value;
    }

    public int Length => RentedArray.Length;

    public RentedArray<T> RentedArray { get; }

    public void Dispose()
    {
        ((IDisposable)RentedArray).Dispose();
    }

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
