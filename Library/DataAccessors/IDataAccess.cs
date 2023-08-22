namespace GradientDescentSharp.DataAccessors;

/// <summary>
/// Basic data access interface
/// </summary>
/// <typeparam name="T"></typeparam>
public interface IDataAccess<T> : IEnumerable<T>
{
    /// <summary>
    /// Source data access length
    /// </summary>
    public int Length { get; }
    ///<inheritdoc/>
    public T this[int index] { get; set; }
}