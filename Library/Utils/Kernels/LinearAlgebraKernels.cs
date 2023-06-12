
using ILGPU;
using ILGPU.Runtime;
using MatrixView = ILGPU.Runtime.ArrayView2D<float, ILGPU.Stride2D.DenseY>;
using VectorView = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;
namespace GradientDescentSharp.Utils.Kernels;

public static class MatrixViewExtensions{

    public static float At(this VectorView v,int index){
        var result = new float[1];
        v.SubView(index,1).CopyToCPU(result);
        return result[0];
    }

    public static void At(this VectorView v,int index,float value){
        var result = new float[1];
        result[0] = value;
        v.SubView(index,1).CopyFromCPU(result);
    }

    /// <summary>
    /// Reads one value from matrix, by copying it to cpu memory
    /// </summary>
    public static float At(this MatrixView m,int row, int column){
        var result = new float[1,1];
        m.SubView((row,column),(1,1)).AsGeneral().CopyToCPU(result);
        return result[0,0];
    }

    /// <summary>
    /// Sets one value from matrix, by copying it to cpu memory
    /// </summary>
    public static void At(this MatrixView m,int row, int column,float value){
        var result = new float[1,1];
        result[0,0] = value;
        m.SubView((row,column),(1,1)).AsGeneral().CopyFromCPU(result);
    }
}

public class LinearAlgebraProvider
{
    public LinearAlgebraProvider(Accelerator accelerator){
        Accelerator = accelerator;
    }
    /// <summary>
    /// Computes outer product of vector1 and vector2, multiplies it by multiplier and adds to matrix
    /// </summary>
    public Action<Index2D, MatrixView, VectorView, VectorView, float> AddOuterProduct =>
        Accelerator
        .LoadAutoGroupedStreamKernel<Index2D, MatrixView, VectorView, VectorView, float>(AddOuterProductKernel);
    /// <summary>
    /// Index=(m1.Rows,m2.Columns) - result matrix sizes
    /// </summary>
    public Action<Index2D, MatrixView, MatrixView, MatrixView> MatrixMul =>
        Accelerator
        .LoadAutoGroupedStreamKernel<Index2D, MatrixView, MatrixView, MatrixView>(MatrixMulKernel);
    /// <summary>
    /// Right side matrix/vector multiplication<br/>
    /// result = matrix*vector<br/>
    /// Index=result vec length<br/>
    /// First = matrix<br/>
    /// Second = right side vector<br/>
    /// Third = result vector<br/>
    /// </summary>
    public Action<Index1D, MatrixView, VectorView, VectorView> MatrixVectorRightSideMul =>
        Accelerator
        .LoadAutoGroupedStreamKernel<Index1D, MatrixView, VectorView, VectorView>(MatrixVectorRightSideMulKernel);
    /// <summary>
    /// Left side matrix/vector multiplication<br/>
    /// result = vector*matrix<br/>
    /// Index=result vec length<br/>
    /// First = matrix that will be treated as transposed<br/>
    /// Second = right side vector<br/>
    /// Third = result vector<br/>
    /// </summary>
    public Action<Index1D, MatrixView, VectorView, VectorView> MatrixVectorLeftSideMul =>
        Accelerator
        .LoadAutoGroupedStreamKernel<Index1D, MatrixView, VectorView, VectorView>(MatrixTransposeVectorRightSideMulKernel);
    /// <summary>
    /// Adds m1 and m2, as m3=m1+multiplier*m2;<br/>
    /// where multiplier is float
    /// </summary>
    public Action<Index2D, MatrixView, MatrixView, float, MatrixView> AddMatricies =>
        Accelerator.
        LoadAutoGroupedStreamKernel<Index2D,MatrixView,MatrixView,float,MatrixView>(AddMatriciesKernel);
    /// <summary>
    /// Adds v1 and v2, as v3=v1+multiplier*v2;<br/>
    /// where multiplier is float
    /// </summary>
    public Action<Index1D, VectorView, VectorView, float, VectorView> AddVectors=>
        Accelerator.
        LoadAutoGroupedStreamKernel<Index1D,VectorView,VectorView,float,VectorView>(AddVectorsKernel);

    public Accelerator Accelerator { get; }

    static void AddMatriciesKernel(Index2D index, MatrixView m1, MatrixView m2,float m2Multiplier, MatrixView result){
        result[index]=m1[index]+m2Multiplier*m2[index];
    }

    static void AddVectorsKernel(Index1D index, VectorView v1, VectorView v2,float v2Multiplier,VectorView result){
        result[index]=v1[index]+v2Multiplier*v2[index];
    }

    static void AddOuterProductKernel(Index2D index, MatrixView matrix, VectorView v1, VectorView v2, float multiplier)
    {
        matrix[index] += multiplier * v1[index.X] * v2[index.Y];
    }
    static void MatrixMulKernel(Index2D index, MatrixView m1, MatrixView m2, MatrixView result)
    {
        var i = index.X;
        var j = index.Y;
        float num = 0f;
        for (int k = 0; k < m1.Extent.Y; k++)
        {
            num += m1[i, k] * m2[k, j];
        }
        result[i, j] = num;
    }
    //index = result length
    static void MatrixVectorRightSideMulKernel(Index1D index, MatrixView mat, VectorView rightSide, VectorView result)
    {
        float num = 0f;
        for (int j = 0; j < mat.Extent.Y; j++)
        {
            num += mat[index, j] * rightSide[j];
        }
        result[index] = num;
    }

    static void MatrixTransposeVectorRightSideMulKernel(Index1D index, MatrixView mat, VectorView rightSide, VectorView result)
    {
        float num = 0f;
        for (int j = 0; j < mat.Extent.X; j++)
        {
            num += mat[j,index] * rightSide[j];
        }
        result[index] = num;
    }
}