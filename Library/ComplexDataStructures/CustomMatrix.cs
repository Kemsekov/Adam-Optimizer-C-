using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
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

/// <summary>
/// Custom matrix
/// </summary>
public class CustomMatrixSingle : MathNet.Numerics.LinearAlgebra.Single.Matrix
{
    ///<inheritdoc/>
    public CustomMatrixSingle(MatrixStorage<float> storage) : base(storage)
    {
    }
}
