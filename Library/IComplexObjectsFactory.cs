namespace GradientDescentSharp;

/// <summary>
/// Factory object that can be used to create linear algebra objects, like vectors and matrices
/// </summary>
public interface IComplexObjectsFactory{
    Vector CreateVector(int length);
    Matrix CreateMatrix(int rows, int columns);
}
