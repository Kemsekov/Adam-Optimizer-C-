using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AdamOptimizer;
using MathNet.Numerics.LinearAlgebra.Storage;

public class CustomMatrix : Matrix
{
    public CustomMatrix(MatrixStorage<double> storage) : base(storage)
    {
    }
}

class CustomMatrixStorage : MatrixStorage<double>
{
    public override bool IsDense => true;

    public override bool IsFullyMutable => true;

    public int StartIndex { get; }
    public IDataAccess<double> Data { get; }

    public CustomMatrixStorage(IDataAccess<double> data,int startIndex, int rowCount, int columnCount) : base(rowCount,columnCount){
        StartIndex = startIndex;
        Data = data;
    }
    int computeIndex(int row, int column){
        return StartIndex+column+row*ColumnCount;
    }
    public override double At(int row, int column)
    {
        return Data[computeIndex(row,column)];
    }

    public override void At(int row, int column, double value)
    {
        Data[computeIndex(row,column)] = value;
    }

    public override bool IsMutableAt(int row, int column)
    {
        return true;
    }
}