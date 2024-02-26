using System.Diagnostics;
using Tensornet;

namespace GradientDescentSharp.Utils;

/// <summary>
/// Basic tensor operations extension
/// </summary>
public unsafe static class TensorExtensions
{
    public static Stopwatch Timer = new();
    /// <summary>
    /// Delegate that allows to do tensor map with span as parameter
    /// </summary>
    public delegate T MapWithSpan<T>(int index,Span<T> span, T value);
    /// <summary>
    /// Delegate that allows to do tensor map with two spans as parameter
    /// </summary>
    public delegate T MapWith2Span<T>(int i,int j,Span<T> span1,Span<T> span2, T value);

    /// <summary>
    /// Map inplace 2d tensor
    /// </summary>
    public static void VecMapInplace<T>(this Tensor<T> tensor, Func<int, T, T> map)
    where T : unmanaged, IEquatable<T>, IConvertible
    {
        Timer.Start();
        var span = tensor.AsSpan();
        var len = span.Length;
        // ref var pos = ref span[0];
        fixed(T* p = &span[0])
        for (int i = 0; i < len; i++){
            ref var pos = ref p[i];
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
        // ref var pos = ref span[0];
        fixed(T* p = &span[0])
        for (int i = 0; i < len; i++){
            ref var pos = ref p[i];
            pos = map(i,inputSpan, pos);
        }
        Timer.Stop();
    }
    /// <summary>
    /// Map inplace 2d tensor
    /// </summary>
    public static void MapInplace<T>(this Tensor<T> tensor, Func<int,int, T, T> map)
    where T : unmanaged, IEquatable<T>, IConvertible
    {
        Timer.Start();
        var span = tensor.AsSpan();
        var rows = tensor.Shape[0];
        var cols = tensor.Shape[1];
        // ref T pos = ref span[0];
        fixed(T* p = &span[0])
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                ref var pos = ref p[i * cols + j];
                pos = map(i, j , pos);
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
        // ref T pos = ref span[0];
        fixed(T* p = &span[0])
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                ref var pos = ref p[i * cols + j];
                pos = map(i, j ,span1,span2, pos);
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
        fixed(T* tensorP = &span[0])
        fixed(T* resP = &resSpan[0])
        for (int i = 0; i < len; i++){
            resP[i] = map(i, tensorP[i]);
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
        fixed(T* tensorP = &span[0])
        fixed(T* resP = &resSpan[0])
        for (int i = 0; i < len; i++){
            resP[i] = map(i,inputSpan, tensorP[i]);
        }
        Timer.Stop();
        return result;
    }
    /// <summary>
    /// Map 2d tensor to new tensor
    /// </summary>
    public static Tensor<T> Map<T>(this Tensor<T> tensor, Func<int,int, T, T> map)
    where T : unmanaged, IEquatable<T>, IConvertible
    {
        Timer.Start();
        var result = Tensor.ZerosLike<T,T>(tensor);
        var span = tensor.AsSpan();
        var resSpan = result.AsSpan();

        var rows = tensor.Shape[0];
        var cols = tensor.Shape[1];
        int pos = 0;
        fixed(T* tensorP = &span[0])
        fixed(T* resP = &resSpan[0])
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                pos = i * cols + j;
                resP[pos] = map(i, j , tensorP[pos]);
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