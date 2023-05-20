using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Storage;
namespace GradientDescentSharp.ComplexDataStructures;
/// <summary>
/// Custom matrix
/// </summary>
public class CustomMatrix : Matrix
{
    ///<inheritdoc/>
    public CustomMatrix(MatrixStorage<double> storage) : base(storage)
    {
    }
}
