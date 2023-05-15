namespace GradientDescentSharp.DataAccessors;

/// <summary>
/// Basic data access interface
/// </summary>
/// <typeparam name="T"></typeparam>
public interface IDataAccess<T> : IEnumerable<T>
{
    public int Length { get; }
    public T this[int index] { get; set; }
}