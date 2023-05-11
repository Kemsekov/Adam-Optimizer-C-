namespace GradientDescentSharp.DataAccessors;

public interface IDataAccess<T> : IEnumerable<T>
{
    public int Length { get; }
    public T this[int index] { get; set; }
}