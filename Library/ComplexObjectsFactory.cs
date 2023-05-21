using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace GradientDescentSharp;

public class ComplexObjectsFactory : IComplexObjectsFactory<double>
{
    public ComplexObjectsFactory(IDataAccess<double> data)
    {
        Data = data;
    }
    public IDataAccess<double> Data { get; }
    int CurrentPosition = 0;
    public Vector<double> CreateVector(int length){
        var storage = new DataAccessVectorStorage(Data,CurrentPosition,length);
        var res = new CustomVector(storage);
        CurrentPosition+=length;
        ThrowIfOutOfRange();
        return res;
    }
    public Matrix<double> CreateMatrix(int rows, int columns){
        var storage = new DataAccessMatrixStorage(Data,CurrentPosition,rows,columns);
        var res = new CustomMatrix(storage);
        CurrentPosition+=rows*columns;
        ThrowIfOutOfRange();
        return res;
    }
    private void ThrowIfOutOfRange()
    {
        if(CurrentPosition>Data.Length)
            throw new Exception($"In ComplexObjectsFactory you took too much objects.\n So much that there is no space left in underlying IDataAccess.\n {CurrentPosition}>{Data.Length}");
    }
}