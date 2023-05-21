using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Storage;
namespace GradientDescentSharp.ComplexDataStructures;

/// <summary>
/// Parallelized matrix storage
/// </summary>
public class ParallelMatrixStorage<T> : MatrixStorage<T>
where T : unmanaged, IFormattable, System.IEquatable<T>
{
    MatrixStorage<T> m;
    /// <summary>
    /// Only works for thread safe storages
    /// </summary>
    public ParallelMatrixStorage(MatrixStorage<T> threadsafeStorage) : base(threadsafeStorage.RowCount, threadsafeStorage.ColumnCount)
    {
        this.m = threadsafeStorage;
    }

    public override bool IsDense => m.IsDense;

    public override bool IsFullyMutable => m.IsFullyMutable;

    public override T At(int row, int column)
    {
        return m.At(row, column);
    }

    public override void At(int row, int column, T value)
    {
        m.At(row, column, value);
    }

    public override bool IsMutableAt(int row, int column)
    {
        return m.IsMutableAt(row, column);
    }
    public override void MapInplace(Func<T, T> f, Zeros zeros)
    {
        Parallel.For(0L,1L*RowCount*ColumnCount,relationalIndex=>{
            var i = (int)relationalIndex/ColumnCount;
            var j = (int)relationalIndex%ColumnCount;
            At(i, j, f(At(i, j)));
        });
    }
    public override void MapIndexedInplace(Func<int, int, T, T> f, Zeros zeros)
    {
        Parallel.For(0L,1L*RowCount*ColumnCount,relationalIndex=>{
            var i = (int)relationalIndex/ColumnCount;
            var j = (int)relationalIndex%ColumnCount;
            At(i, j, f(i, j, At(i, j)));
        });
    }
}
