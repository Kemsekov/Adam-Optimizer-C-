using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace GradientDescentSharp;

///<inheritdoc/>
public class ComplexObjectsFactory : IComplexObjectsFactory<double>
{
    /// <summary>
    /// Creates new instance of <see cref="ComplexObjectsFactory"/> 
    /// </summary>
    public ComplexObjectsFactory(IDataAccess<double> data)
    {
        Data = data;
    }
    /// <summary>
    /// Data source that is used to divide it into parts
    /// </summary>
    public IDataAccess<double> Data { get; }
    int CurrentPosition = 0;
    /// <summary>
    /// Create vector of given length
    /// </summary>
    public DVector CreateVector(int length){
        var storage = new DataAccessVectorStorage(Data,CurrentPosition,length);
        var res = new CustomVector(storage);
        CurrentPosition+=length;
        ThrowIfOutOfRange();
        return res;
    }
    /// <summary>
    /// Create matrix of given size
    /// </summary>
    public DMatrix CreateMatrix(int rows, int columns){
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

///<inheritdoc/>
public class ComplexObjectsFactorySingle : IComplexObjectsFactory<float>
{
    /// <summary>
    /// Creates new instance of <see cref="ComplexObjectsFactorySingle"/> 
    /// </summary>
    public ComplexObjectsFactorySingle(IDataAccess<float> data)
    {
        Data = data;
    }
    /// <summary>
    /// Data source that is used to divide it into parts
    /// </summary>
    public IDataAccess<float> Data { get; }
    int CurrentPosition = 0;
    /// <summary>
    /// Create vector of given length
    /// </summary>
    public FVector CreateVector(int length){
        var storage = new DataAccessVectorStorageSingle(Data,CurrentPosition,length);
        var res = new CustomVectorSingle(storage);
        CurrentPosition+=length;
        ThrowIfOutOfRange();
        return res;
    }
    /// <summary>
    /// Create matrix of given size
    /// </summary>
    public FMatrix CreateMatrix(int rows, int columns){
        var storage = new DataAccessMatrixStorageSingle(Data,CurrentPosition,rows,columns);
        var res = new CustomMatrixSingle(storage);
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
