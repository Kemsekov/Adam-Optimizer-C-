namespace GradientDescentSharp;

public class ComplexObjectsFactory : IComplexObjectsFactory
{
    public ComplexObjectsFactory(IDataAccess<double> data)
    {
        Data = data;
    }
    public IDataAccess<double> Data { get; }
    int CurrentPosition = 0;
    public Vector CreateVector(int length){
        var res = new CustomVector(new CustomVectorStorage(Data,CurrentPosition,length));
        CurrentPosition+=length;
        ThrowIfOutOfRange();
        return res;
    }
    public Matrix CreateMatrix(int rows, int columns){
        var res = new CustomMatrix(new CustomMatrixStorage(Data,CurrentPosition,rows,columns));
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