using MathNet.Numerics.LinearAlgebra.Storage;
namespace GradientDescentSharp.ComplexDataStructures;

class DataAccessMatrixStorage : MatrixStorage<double>
{
    public override bool IsDense => true;

    public override bool IsFullyMutable => true;

    public int StartIndex { get; }
    public IDataAccess<double> Data { get; }

    public DataAccessMatrixStorage(IDataAccess<double> data, int startIndex, int rowCount, int columnCount) : base(rowCount, columnCount)
    {
        StartIndex = startIndex;
        Data = data;
    }
    int computeIndex(int row, int column)
    {
        return StartIndex + column + row * ColumnCount;
    }
    public override double At(int row, int column)
    {
        return Data[computeIndex(row, column)];
    }

    public override void At(int row, int column, double value)
    {
        Data[computeIndex(row, column)] = value;
    }

    public override bool IsMutableAt(int row, int column)
    {
        return true;
    }

}

class DataAccessMatrixStorageSingle : MatrixStorage<float>
{
    public override bool IsDense => true;

    public override bool IsFullyMutable => true;

    public int StartIndex { get; }
    public IDataAccess<float> Data { get; }

    public DataAccessMatrixStorageSingle(IDataAccess<float> data, int startIndex, int rowCount, int columnCount) : base(rowCount, columnCount)
    {
        StartIndex = startIndex;
        Data = data;
    }
    int computeIndex(int row, int column)
    {
        return StartIndex + column + row * ColumnCount;
    }
    public override float At(int row, int column)
    {
        return Data[computeIndex(row, column)];
    }

    public override void At(int row, int column, float value)
    {
        Data[computeIndex(row, column)] = value;
    }

    public override bool IsMutableAt(int row, int column)
    {
        return true;
    }

}