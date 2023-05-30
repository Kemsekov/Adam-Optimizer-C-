namespace GradientDescentSharp.Utils;
public static class DataAccessExtensions
{
    /// <summary>
    /// Applies a function to each element of data, and replaces it with new element
    /// </summary>
    public static void MapInplace<T>(this IDataAccess<T> data,Func<T,T> map){
        for(int i = 0;i<data.Length;i++){
            data[i] = map(data[i]);
        }
    }
}

