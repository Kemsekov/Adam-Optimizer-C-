using System.Collections.Concurrent;

namespace GradientDescentSharp.Utils;

/// <summary>
/// Simple object pool
/// </summary>
public class ObjectPool<T> where T : new()
{
    private readonly ConcurrentBag<T> _objects = new ConcurrentBag<T>();
    private readonly Func<T> _objectGenerator;
    /// <summary>
    /// Creates new object pool
    /// </summary>
    public ObjectPool(Func<T> objectGenerator)
    {
        _objectGenerator = objectGenerator ?? throw new ArgumentNullException(nameof(objectGenerator));
    }
    /// <returns>
    /// Object from pool
    /// </returns>
    public T GetObject()
    {
        if (_objects.TryTake(out T item)) return item;

        return _objectGenerator();
    }
    /// <summary>
    /// Puts item into pool
    /// </summary>
    public void PutObject(T item)
    {
        _objects.Add(item);
    }
}