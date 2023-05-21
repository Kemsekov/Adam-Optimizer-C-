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
    protected override void DoMultiply(Vector<double> rightSide, Vector<double> result)
    {
        for (int j = 0; j < base.ColumnCount; j++)
        {
            var vecj = rightSide[j];
            if(vecj==0) continue;
            for (int i = 0; i < base.RowCount; i++)
            {
                result[i] += base.At(i, j) * vecj;
            }
        }
    }
}
