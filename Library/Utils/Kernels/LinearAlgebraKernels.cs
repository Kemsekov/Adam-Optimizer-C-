
using ILGPU;
using ILGPU.Runtime;
using MatrixView = ILGPU.Runtime.ArrayView2D<float, ILGPU.Stride2D.DenseY>;
using VectorView = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;
namespace GradientDescentSharp.Utils.Kernels;

public record DisposableLinearAlgebraProvider(Context Context, Accelerator Accelerator, LinearAlgebraProvider Provider) : IDisposable
{
    ~DisposableLinearAlgebraProvider(){
        Dispose();
    }
    public void Dispose()
    {
        Accelerator.Dispose();
        Context.Dispose();
    }
}

public class LinearAlgebraProvider
{
    /// <summary>
    /// Creates a context
    /// </summary>
    public static DisposableLinearAlgebraProvider Create(bool preferCpu = false){
        var context = Context.CreateDefault();
        var accelerator = context.GetPreferredDevice(preferCpu).CreateAccelerator(context);
        return new(context,accelerator,new(accelerator));
    }
    /// <summary>
    /// Compiles all kernels
    /// </summary>
    public static void CompileAllKernels(){
        using var context = Context.CreateDefault();
        using var accelerator = context.GetPreferredDevice(false).CreateAccelerator(context);
        var provider = new LinearAlgebraProvider(accelerator);
        var kernels = new dynamic[]{
            provider.AddOuterProductLauncher,
            provider.MatrixMulLauncher,
            provider.MatrixVectorRightSideMulLauncher,
            provider.MatrixVectorLeftSideMulLauncher,
            provider.AddMatricesLauncher,
            provider.AddVectorsLauncher,
            provider.DotLauncher
        };
    }
    public LinearAlgebraProvider(Accelerator accelerator)
    {
        Accelerator = accelerator;
    }

    protected Action<Index1D, MatrixView, VectorView, VectorView, float> AddOuterProductLauncher =>
        Accelerator
        .LoadAutoGroupedStreamKernel<Index1D, MatrixView, VectorView, VectorView, float>(AddOuterProductKernel);
    /// <summary>
    /// Index=(m1.Rows,m2.Columns) - result matrix sizes
    /// </summary>
    protected Action<Index2D, MatrixView, MatrixView, MatrixView> MatrixMulLauncher =>
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
    protected Action<Index1D, MatrixView, VectorView, VectorView> MatrixVectorRightSideMulLauncher =>
        Accelerator
        .LoadAutoGroupedStreamKernel<Index1D, MatrixView, VectorView, VectorView>(MatrixVectorRightSideMulKernel);

    protected Action<Index1D, MatrixView, VectorView, VectorView> MatrixVectorLeftSideMulLauncher =>
        Accelerator
        .LoadAutoGroupedStreamKernel<Index1D, MatrixView, VectorView, VectorView>(MatrixTransposeVectorRightSideMulKernel);

    protected Action<Index1D, MatrixView, MatrixView, float, MatrixView> AddMatricesLauncher =>
        Accelerator.
        LoadAutoGroupedStreamKernel<Index1D, MatrixView, MatrixView, float, MatrixView>(AddMatricesKernel);

    protected Action<Index1D, VectorView, VectorView, float, VectorView,int> AddVectorsLauncher =>
        Accelerator.
        LoadAutoGroupedStreamKernel<Index1D, VectorView, VectorView, float, VectorView,int>(AddVectorsKernel);

    protected Action<Index1D, int, VectorView, VectorView, VectorView> DotLauncher =>
        Accelerator.
        LoadAutoGroupedStreamKernel<Index1D, int, VectorView, VectorView, VectorView>(DotKernel);

    public Accelerator Accelerator { get; }
    /// <summary>
    /// Adds m1 and m2, as result=m1+multiplier*m2;<br/>
    /// where multiplier is float
    /// </summary>
    public void AddMatrices(MatrixView m1, MatrixView m2, float multiplier, MatrixView result){
        AddMatricesLauncher((Index1D)result.Extent.X, m1, m2, multiplier, result);
    }
    /// <summary>
    /// Adds v1 and v2, as result=v1+multiplier*v2;<br/>
    /// where multiplier is float <br/>
    /// stepLength is how many numbers of result vector to process on single work unit
    /// </summary>
    public void AddVectors(VectorView v1, VectorView v2, float multiplier, VectorView result,int stepLength=-1){
        if(stepLength<0)
            stepLength=(int)Math.Sqrt(v1.Extent);
        AddVectorsLauncher((Index1D)result.Extent,v1,v2,multiplier,result,stepLength);
    }
    /// <summary>
    /// result=m*v
    /// </summary>
    public void MatrixVectorMul(MatrixView m, VectorView v, VectorView result)
    {
        MatrixVectorRightSideMulLauncher((int)result.Extent, m, v, result);
    }
    /// <summary>
    /// result=v*m
    /// </summary>
    public void MatrixVectorMul(VectorView v, MatrixView m, VectorView result)
    {
        MatrixVectorLeftSideMulLauncher((int)result.Extent, m, v, result);
    }
    /// <summary>
    /// result=m1*m2
    /// </summary>
    public void MatrixMul(MatrixView m1, MatrixView m2, MatrixView result)
    {
        MatrixMulLauncher((Index2D)result.Extent, m1, m2, result);
    }
    /// <summary>
    /// Computes outer product of vector1 and vector2, multiplies it by multiplier and adds to matrix
    /// </summary>
    public void AddOuterProduct(MatrixView m1, VectorView v1, VectorView v2, float multiplier)
    {
        AddOuterProductLauncher((Index1D)m1.Extent.X, m1, v1, v2, multiplier);
    }
    /// <summary>
    /// Divides a vectors into "stepLength" chunks and compute dot product in each chunk.<br/>
    /// </summary>
    /// <param name="stepLength">Will be set automatically if left to -1</param>
    /// <returns></returns>
    public float Dot(VectorView v1, VectorView v2, int stepLength = -1)
    {
        if(stepLength<0)
            stepLength=(int)Math.Sqrt(v1.Extent);
        
        var size = v1.Extent / stepLength;
        size = size == 0 ? 1 : size;
        using var mapReduce = Accelerator.Allocate1D<float>(size);
        DotLauncher((Index1D)size, stepLength, v1, v2, mapReduce);
        var tmp = new float[size];
        mapReduce.CopyToCPU(tmp);
        return tmp.Sum();
    }
    static void DotKernel(Index1D index, int stepLength, VectorView v1, VectorView v2, VectorView mapReduceResult)
    {
        var midSum = 0.0f;
        var i = index * stepLength;
        var end = i + stepLength;
        if (v1.Extent - end < stepLength)
            end = (Index1D)v1.Extent;

        for (; i < end; i++)
        {
            midSum += v1[i] * v2[i];
        }

        mapReduceResult[index] = midSum;
    }
    /// <summary>
    /// Because we store matrices in DenseY stride, it is more preferable to iterate
    /// over Y values in a single work unit to get max gpu cache hits
    /// </summary>
    static void AddMatricesKernel(Index1D xIndex, MatrixView m1, MatrixView m2, float m2Multiplier, MatrixView result)
    {
        var ySize = result.Extent.Y;
        for(int y = 0;y<ySize;y++){
            result[xIndex,y] = m1[xIndex,y] + m2Multiplier * m2[xIndex,y];
        }
    }
    static void AddVectorsKernel(Index1D index, VectorView v1, VectorView v2, float v2Multiplier, VectorView result,int stepLength)
    {
        var i = index * stepLength;
        var end = i + stepLength;
        if (v1.Extent - end < stepLength)
            end = (Index1D)v1.Extent;

        for (; i < end; i++)
        {
            result[i]= v1[i] + v2Multiplier*v2[i];
        }
    }
    static void AddOuterProductKernel(Index1D xIndex, MatrixView matrix, VectorView v1, VectorView v2, float multiplier)
    {
        var ySize = matrix.Extent.Y;
        var v1XValue=v1[xIndex];
        for(int y = 0;y<ySize;y++)
            matrix[xIndex,y] += multiplier * v1XValue * v2[y];
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
            num += mat[j, index] * rightSide[j];
        }
        result[index] = num;
    }
}