using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using MathNet.Numerics.LinearAlgebra.Storage;
namespace GradientDescentSharp.NeuralNetwork;
/// <summary>
/// Custom matrix for neural networks, optimized for faster multiplications
/// </summary>
public class NNMatrix : Matrix
{
    ///<inheritdoc/>
    public NNMatrix(MatrixStorage<float> storage) : base(storage)
    {
    }

    public NNMatrix(int rows, int columns, float value) : base(DenseMatrix.Create(rows,columns,value).Storage)
    {
    }
    protected override void DoMultiply(Vector<float> rightSide, Vector<float> result)
    {
        // this matrix multiplication reverses order of multiplication to
        // reduce memory reading and skip zero values.
        // It is helpful for classification tasks, where a lot of vector values is zeros
        for (int j = 0; j < ColumnCount; j++)
        {
            var vecj = rightSide[j];
            if(vecj==0) continue;
            for (int i = 0; i < RowCount; i++)
            {
                result[i] += At(i, j) * vecj;
            }
        }
    }
    public static NNMatrix Create(int rows, int columns, float value)
    {
        return new NNMatrix(rows, columns,value);
    }
    public static NNMatrix Create(int rows, int columns, Func<int, int, float> init)
    {
        var mat = new NNMatrix(rows, columns,0);
        mat.MapIndexedInplace((j,k,x)=>init(j,k));
        return mat;
    }
}
