using ILGPU.Runtime;
using MatrixView = ILGPU.Runtime.ArrayView2D<float, ILGPU.Stride2D.DenseY>;
using VectorView = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;
namespace GradientDescentSharp.Utils.Kernels;

public static class MatrixViewExtensions
{

    public static float At(this VectorView v, int index)
    {
        var result = new float[1];
        v.SubView(index, 1).CopyToCPU(result);
        return result[0];
    }

    public static void At(this VectorView v, int index, float value)
    {
        var result = new float[1];
        result[0] = value;
        v.SubView(index, 1).CopyFromCPU(result);
    }

    /// <summary>
    /// Reads one value from matrix, by copying it to cpu memory
    /// </summary>
    public static float At(this MatrixView m, int row, int column)
    {
        var result = new float[1, 1];
        m.SubView((row, column), (1, 1)).AsGeneral().CopyToCPU(result);
        return result[0, 0];
    }

    /// <summary>
    /// Sets one value from matrix, by copying it to cpu memory
    /// </summary>
    public static void At(this MatrixView m, int row, int column, float value)
    {
        var result = new float[1, 1];
        result[0, 0] = value;
        m.SubView((row, column), (1, 1)).AsGeneral().CopyFromCPU(result);
    }
}
