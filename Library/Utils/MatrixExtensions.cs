namespace GradientDescentSharp.Utils;
public static class MatrixExtensions
{
    public static IEnumerable<double> ToEnumerable(this Matrix m){
        for(int r = 0;r<m.RowCount;r++){
            for(int k = 0;k<m.ColumnCount;k++){
                yield return m[r,k];
            }
        }
    }
}