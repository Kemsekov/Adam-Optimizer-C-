using MathNet.Numerics.LinearAlgebra.Storage;
namespace GradientDescentSharp.ComplexDataStructures;
/// <summary>
/// Custom matrix that store it's values on data accessors that can be used by gradient descent
/// </summary>
public class CustomMatrix : Matrix
{
    ///<inheritdoc/>
    public CustomMatrix(MatrixStorage<FloatType> storage) : base(storage)
    {
    }
}
class CustomMatrixStorage : MatrixStorage<FloatType>
{
    public override bool IsDense => true;

    public override bool IsFullyMutable => true;

    public int StartIndex { get; }
    public IDataAccess<FloatType> Data { get; }

    public CustomMatrixStorage(IDataAccess<FloatType> data, int startIndex, int rowCount, int columnCount) : base(rowCount, columnCount)
    {
        StartIndex = startIndex;
        Data = data;
    }
    int computeIndex(int row, int column)
    {
        return StartIndex + column + row * ColumnCount;
    }
    public override FloatType At(int row, int column)
    {
        return Data[computeIndex(row, column)];
    }

    public override void At(int row, int column, FloatType value)
    {
        Data[computeIndex(row, column)] = value;
    }

    public override bool IsMutableAt(int row, int column)
    {
        return true;
    }
}