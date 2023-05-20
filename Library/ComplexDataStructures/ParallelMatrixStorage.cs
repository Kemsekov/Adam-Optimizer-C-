using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Storage;
namespace GradientDescentSharp.ComplexDataStructures;

/// <summary>
/// Parallelized matrix storage
/// </summary>
public class ParallelMatrixStorage : MatrixStorage<double>{
    MatrixStorage<double> m;
    public ParallelMatrixStorage(MatrixStorage<double> m) : base(m.RowCount, m.ColumnCount)
    {
        this.m = m;
    }

    public override bool IsDense => m.IsDense;

    public override bool IsFullyMutable => m.IsFullyMutable;

    public override double At(int row, int column)
    {
        return m.At(row, column);
    }

    public override void At(int row, int column, double value)
    {
        m.At(row, column, value);
    }

    public override bool IsMutableAt(int row, int column)
    {
        return m.IsMutableAt(row, column);
    }
    public override void MapInplace(Func<double, double> f, Zeros zeros)
    {
        Parallel.For(0L,1L*RowCount*ColumnCount,relationalIndex=>{
            var i = (int)relationalIndex/ColumnCount;
            var j = (int)relationalIndex%ColumnCount;
            At(i, j, f(At(i, j)));
        });
    }
    public override void MapIndexedInplace(Func<int, int, double, double> f, Zeros zeros)
    {
        Parallel.For(0L,1L*RowCount*ColumnCount,relationalIndex=>{
            var i = (int)relationalIndex/ColumnCount;
            var j = (int)relationalIndex%ColumnCount;
            At(i, j, f(i, j, At(i, j)));
        });
    }
}
