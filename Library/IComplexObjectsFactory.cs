using MathNet.Numerics.LinearAlgebra;
namespace GradientDescentSharp;

/// <summary>
/// Factory object that can be used to create linear algebra objects, like vectors and matrices
/// </summary>
public interface IComplexObjectsFactory<T>
where T : unmanaged, IFormattable, System.IEquatable<T>
{
    /// <summary>
    /// Creates a vector of given length
    /// </summary>
    Vector<T> CreateVector(int length);
    /// <summary>
    /// Creates a matrix of given size
    /// </summary>
    Matrix<T> CreateMatrix(int rows, int columns);
}
