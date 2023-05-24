using System.Collections.Concurrent;

namespace GradientDescentSharp.Utils;

/// <summary>
/// Simple object pool
/// </summary>
public class ObjectPool<T> where T : new()
{
    private readonly ConcurrentBag<T> _objects = new ConcurrentBag<T>();
    private readonly Func<T> _objectGenerator;

    public ObjectPool(Func<T> objectGenerator)
    {
        _objectGenerator = objectGenerator ?? throw new ArgumentNullException(nameof(objectGenerator));
    }

    public T GetObject()
    {
        if (_objects.TryTake(out T item)) return item;

        return _objectGenerator();
    }

    public void PutObject(T item)
    {
        _objects.Add(item);
    }
}