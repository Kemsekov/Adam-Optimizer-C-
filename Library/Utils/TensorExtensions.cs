using System.Diagnostics;
using Tensornet;

namespace GradientDescentSharp.Utils;

/// <summary>
/// Basic tensor operations extension
/// </summary>
public static class TensorExtensions
{
    public static Stopwatch Timer = new();
    /// <summary>
    /// Delegate that allows to do tensor map with span as parameter
    /// </summary>
    public delegate T MapWithSpan<T>(int index,Span<T> span, T value);
    /// <summary>
    /// Delegate that allows to do tensor map with two spans as parameter
    /// </summary>
    public delegate T MapWith2Span<T>(int[] index,Span<T> span1,Span<T> span2, T value);

    /// <summary>
    /// Map inplace 2d tensor
    /// </summary>
    public static void VecMapInplace<T>(this Tensor<T> tensor, Func<int, T, T> map)
    where T : unmanaged, IEquatable<T>, IConvertible
    {
        Timer.Start();
        var span = tensor.AsSpan();
        var len = span.Length;
        for (int i = 0; i < len; i++){
            ref var pos = ref span[i];
            pos = map(i, pos);
        }
        Timer.Stop();

    }
    /// <summary>
    /// Map inplace 2d tensor
    /// </summary>
    public static void VecMapInplace<T>(this Tensor<T> tensor,Span<T> inputSpan, MapWithSpan<T> map)
    where T : unmanaged, IEquatable<T>, IConvertible
    {
        Timer.Start();
        var span = tensor.AsSpan();
        var len = span.Length;
        for (int i = 0; i < len; i++){
            ref var pos = ref span[i];
            pos = map(i,inputSpan, pos);
        }
        Timer.Stop();
    }
    /// <summary>
    /// Map inplace 2d tensor
    /// </summary>
    public static void MapInplace<T>(this Tensor<T> tensor, Func<int[], T, T> map)
    where T : unmanaged, IEquatable<T>, IConvertible
    {
        Timer.Start();
        var span = tensor.AsSpan();
        var rows = tensor.Shape[0];
        var cols = tensor.Shape[1];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                ref var pos = ref span[i * cols + j];
                pos = map(new[] { i, j }, pos);
            }
        Timer.Stop();
    }
    /// <summary>
    /// Map inplace 2d tensor
    /// </summary>
    public static void MapInplace<T>(this Tensor<T> tensor,Span<T> span1, Span<T> span2, MapWith2Span<T> map)
    where T : unmanaged, IEquatable<T>, IConvertible
    {
        Timer.Start();
        var span = tensor.AsSpan();
        var rows = tensor.Shape[0];
        var cols = tensor.Shape[1];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                ref var pos = ref span[i * cols + j];
                pos = map(new[] { i, j },span1,span2, pos);
            }
        Timer.Stop();
    }
    /// <summary>
    /// Map inplace 2d tensor
    /// </summary>
    public static void MapInplace<T>(this Tensor<T> tensor, Func<T, T> map)
    where T : unmanaged, IEquatable<T>, IConvertible
    {
        tensor.VecMapInplace((i,v)=>map(v));
    }
    /// <summary>   
    /// Map 1d tensor to new tensor
    /// </summary>
    public static Tensor<T> VecMap<T>(this Tensor<T> tensor, Func<int, T, T> map)
    where T : unmanaged, IEquatable<T>, IConvertible
    {
        Timer.Start();
        var result = Tensor.ZerosLike<T,T>(tensor);
        var span = tensor.AsSpan();
        var resSpan = result.AsSpan();

        var len = span.Length;
        for (int i = 0; i < len; i++){
            resSpan[i] = map(i, span[i]);
        }
        Timer.Stop();
        return result;
    }
    /// <summary>   
    /// Map 1d tensor to new tensor with additional span
    /// </summary>
    public static Tensor<T> VecMap<T>(this Tensor<T> tensor,Span<T> inputSpan, MapWithSpan<T> map)
    where T : unmanaged, IEquatable<T>, IConvertible
    {
        Timer.Start();
        var result = Tensor.ZerosLike<T,T>(tensor);
        var span = tensor.AsSpan();
        var resSpan = result.AsSpan();

        var len = span.Length;
        for (int i = 0; i < len; i++){
            resSpan[i] = map(i,inputSpan, span[i]);
        }
        Timer.Stop();
        return result;
    }
    /// <summary>
    /// Map 2d tensor to new tensor
    /// </summary>
    public static Tensor<T> Map<T>(this Tensor<T> tensor, Func<int[], T, T> map)
    where T : unmanaged, IEquatable<T>, IConvertible
    {
        Timer.Start();
        var result = Tensor.ZerosLike<T,T>(tensor);
        var span = tensor.AsSpan();
        var resSpan = result.AsSpan();

        var rows = tensor.Shape[0];
        var cols = tensor.Shape[1];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                var pos = i * cols + j;
                resSpan[pos] = map(new[] { i, j }, span[pos]);
            }
        Timer.Stop();
        return result;
    }
    /// <summary>
    /// Map 2d tensor to new tensor
    /// </summary>
    public static Tensor<T> Map<T>(this Tensor<T> tensor, Func<T, T> map)
    where T : unmanaged, IEquatable<T>, IConvertible
    {
        return tensor.VecMap((i,v)=>map(v));
    }
    /// <summary>
    /// Read 2d tensor at position - it is a bit faster than original tensor[a,b]
    /// </summary>
    public static ref T VecAt<T>(this Tensor<T> tensor, int pos)
    where T : unmanaged, IEquatable<T>, IConvertible
    {
        return ref tensor.AsSpan()[pos];
    }
    /// <summary>
    /// Read 2d tensor at position - it is a bit faster than original tensor[a,b]
    /// </summary>
    public static ref T At<T>(this Tensor<T> tensor, int[] pos)
    where T : unmanaged, IEquatable<T>, IConvertible
    {
        return ref tensor.AsSpan()[pos[0] * tensor.Shape[0] + pos[1]];
    }
}