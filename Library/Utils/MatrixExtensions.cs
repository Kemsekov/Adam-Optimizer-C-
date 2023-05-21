using MathNet.Numerics.LinearAlgebra;
namespace GradientDescentSharp.Utils;
public static class MatrixExtensions
{
    public static IEnumerable<T> ToEnumerable<T>(this Matrix<T> m)
    where T : unmanaged, IFormattable, System.IEquatable<T>
    {
        for(int r = 0;r<m.RowCount;r++){
            for(int k = 0;k<m.ColumnCount;k++){
                yield return m[r,k];
            }
        }
    }
}