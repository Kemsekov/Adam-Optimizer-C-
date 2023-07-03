
using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using MatrixView = ILGPU.Runtime.ArrayView2D<float, ILGPU.Stride2D.DenseY>;
using VectorView = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;
using VectorBuffer = ILGPU.Runtime.MemoryBuffer1D<float, ILGPU.Stride1D.Dense>;
using MatrixBuffer = ILGPU.Runtime.MemoryBuffer2D<float, ILGPU.Stride2D.DenseY>;
namespace GradientDescentSharp.Utils.Kernels;

public record DisposableLinearAlgebraProvider(Context Context, Accelerator Accelerator, LinearAlgebraProvider Provider) : IDisposable
{
    ~DisposableLinearAlgebraProvider()
    {
        Dispose();
    }
    public void Dispose()
    {
        Accelerator.Dispose();
        Context.Dispose();
    }
}

public unsafe class LinearAlgebraProvider
{
    /// <summary>
    /// Creates a context
    /// </summary>
    public static DisposableLinearAlgebraProvider Create(bool preferCpu = false)
    {
        var context = Context.Create(b=>{
            b.Optimize(OptimizationLevel.Release);
            b.EnableAlgorithms();
            b.Default();
        });
        var accelerator = context.GetPreferredDevice(preferCpu).CreateAccelerator(context);
        return new(context, accelerator, new(accelerator));
    }
    /// <summary>
    /// Compiles all kernels
    /// </summary>
    public static void CompileAllKernels(bool preferCpu = false)
    {
        using var source = LinearAlgebraProvider.Create(preferCpu);
        var provider = source.Provider;
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

    protected Action<Index1D, VectorView, VectorView, float, VectorView, int> AddVectorsLauncher =>
        Accelerator.
        LoadAutoGroupedStreamKernel<Index1D, VectorView, VectorView, float, VectorView, int>(AddVectorsKernel);

    protected Action<Index1D, int, VectorView, VectorView, VectorView> DotLauncher =>
        Accelerator.
        LoadAutoGroupedStreamKernel<Index1D, int, VectorView, VectorView, VectorView>(DotKernel);
    protected Action<Index1D, int, MatrixView,int, VectorView> SetRowLauncher =>
        Accelerator.
        LoadAutoGroupedStreamKernel<Index1D, int, MatrixView,int, VectorView>(SetMatrixRowKernel);
    protected Action<Index1D, int, MatrixView,int, VectorView> SetColumnLauncher =>
        Accelerator.
        LoadAutoGroupedStreamKernel<Index1D, int, MatrixView,int, VectorView>(SetMatrixColumnKernel);
    protected Action<Index1D, int, MatrixView,int, VectorView> CopyRowLauncher =>
        Accelerator.
        LoadAutoGroupedStreamKernel<Index1D, int, MatrixView,int, VectorView>(CopyMatrixRowKernel);
    protected Action<Index1D, int, MatrixView,int, VectorView> CopyColumnLauncher =>
        Accelerator.
        LoadAutoGroupedStreamKernel<Index1D, int, MatrixView,int, VectorView>(CopyMatrixColumnKernel);
    protected Action<Index1D, MatrixView,VectorView> MatrixL2Launcher =>
        Accelerator.
        LoadAutoGroupedStreamKernel<Index1D, MatrixView,VectorView>(MatrixL2Kernel);

    public Accelerator Accelerator { get; }
    /// <returns>
    /// Vector of given size. DO NOT FORGET TO DISPOSE IT!
    /// </returns>
    public VectorBuffer CreateVector(int length)
        =>Accelerator.Allocate1D<float>(length);
    /// <returns>
    /// Matrix of given size. DO NOT FORGET TO DISPOSE IT!
    /// </returns>
    public MatrixBuffer CreateMatrix(int rows, int columns)
        =>Accelerator.Allocate2DDenseY<float>((rows,columns));
    public void SetRow(MatrixView matrix, int row, VectorView source){
        var stepLength = DetermineStepLength(source, -1);
        SetRowLauncher(DetermineStepsCount(source,stepLength),stepLength,matrix,row,source);
    }
    public void SetColumn(MatrixView matrix, int column, VectorView source){
        var stepLength = DetermineStepLength(source, -1);
        SetColumnLauncher(DetermineStepsCount(source,stepLength),stepLength,matrix,column,source);
    }
    public void CopyRow(MatrixView matrix, int row, VectorView result){
        var stepLength = DetermineStepLength(result, -1);
        CopyRowLauncher(DetermineStepsCount(result,stepLength),stepLength,matrix,row,result);
    }
    public void CopyColumn(MatrixView matrix, int column, VectorView result){
        var stepLength = DetermineStepLength(result, -1);
        CopyColumnLauncher(DetermineStepsCount(result,stepLength),stepLength,matrix,column,result);
    }
    /// <summary>
    /// Adds m1 and m2, as result=m1+multiplier*m2;<br/>
    /// where multiplier is float
    /// </summary>
    public void AddMatrices(MatrixView m1, MatrixView m2, float multiplier, MatrixView result)
    {
        AddMatricesLauncher((Index1D)result.Extent.X, m1, m2, multiplier, result);
    }
    /// <summary>
    /// Adds v1 and v2, as result=v1+multiplier*v2;<br/>
    /// where multiplier is float <br/>
    /// stepLength is how many numbers of result vector to process on single work unit
    /// </summary>
    public void AddVectors(VectorView v1, VectorView v2, float multiplier, VectorView result, int stepLength = -1)
    {
        stepLength = DetermineStepLength(v1, stepLength);
        AddVectorsLauncher(DetermineStepsCount(v1,stepLength), v1, v2, multiplier, result, stepLength);
    }
    Index1D DetermineStepsCount(VectorView result, int stepLength){
        var size = result.Extent / stepLength;
        size = size == 0 ? 1 : size;
        return (Index1D)size;
    }
    int DetermineStepLength(VectorView result, int stepLength)
    {
        if (stepLength < 0)
            stepLength = (int)result.Extent / Accelerator.MaxNumThreadsPerMultiprocessor;
        if(stepLength>=result.Extent)
            stepLength = (int)result.Extent;
        stepLength = stepLength < 4 ? (int)Math.Sqrt(result.Extent) : stepLength;
        return stepLength;
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
    /// <returns>
    /// L2 norm of vector = v1^2+v2^2+v3^2...
    /// </returns>
    public float L2(VectorView vector){
        return Dot(vector,vector);
    }
    /// <returns>
    /// L2 norm of matrix = m11^2+m12^2...+m21^2+m22^2+...
    /// </returns>
    public float L2(MatrixView matrix){
        using var result = Accelerator.Allocate1D<float>(matrix.Extent.X);
        MatrixL2Launcher((Index1D)matrix.Extent.X,matrix,result);
        return result.GetAsArray1D().Sum();
    }
    /// <summary>
    /// Divides a vectors into "stepLength" chunks and compute dot product in each chunk.<br/>
    /// </summary>
    /// <param name="stepLength">Will be set automatically if left to -1</param>
    public float Dot(VectorView v1, VectorView v2, int stepLength = -1)
    {
        stepLength = DetermineStepLength(v2, stepLength);
        var size = DetermineStepsCount(v1,stepLength);
        using var result = Accelerator.Allocate1D<float>(size);
        DotLauncher(size, stepLength, v1, v2, result);
        return result.GetAsArray1D().Sum();
    }
    static void DotKernel(Index1D index, int stepLength, VectorView v1, VectorView v2, VectorView mapReduceResult)
    {
        var midSum = 0.0f;
        Index1D i, end;
        ComputeIndexes(index, stepLength, v1, out i, out end);
        
        for (; i < end; i++)
        {
            midSum += v1[i] * v2[i];
        }
        mapReduceResult[index]=midSum;
    }
    static void SetMatrixColumnKernel(Index1D index, int stepLength, MatrixView m, int column, VectorView source){
        Index1D i, end;
        ComputeIndexes(index, stepLength, source, out i, out end);
        for (; i < end; i++)
        {
            m[i, column]=source[i];
        }
    }
    static void SetMatrixRowKernel(Index1D index, int stepLength, MatrixView m, int row, VectorView source){
        Index1D i, end;
        ComputeIndexes(index, stepLength, source, out i, out end);
        for (; i < end; i++)
        {
            m[row,i]=source[i];
        }
    }
    static void CopyMatrixColumnKernel(Index1D index, int stepLength, MatrixView m, int column, VectorView result)
    {
        Index1D i, end;
        ComputeIndexes(index, stepLength, result, out i, out end);
        for (; i < end; i++)
        {
            result[i] = m[i, column];
        }
    }
    static void ComputeIndexes(Index1D index, int stepLength, VectorView result, out Index1D i, out Index1D end)
    {
        i = index * stepLength;
        end = i + stepLength;
        if (result.Extent-end<stepLength)
            end = (Index1D)result.Extent;
    }
    static void CopyMatrixRowKernel(Index1D index, int stepLength, MatrixView m, int row, VectorView result)
    {
        Index1D i, end;
        ComputeIndexes(index, stepLength, result, out i, out end);
        for (; i < end; i++){
            result[i]=m[row,i];
        }
    }
    /// <summary>
    /// Because we store matrices in DenseY stride, it is more preferable to iterate
    /// over Y values in a single work unit to get max gpu cache hits
    /// </summary>
    static void AddMatricesKernel(Index1D xIndex, MatrixView m1, MatrixView m2, float m2Multiplier, MatrixView result)
    {
        var ySize = result.Extent.Y;
        for (int y = 0; y < ySize; y++)
        {
            result[xIndex, y] = m1[xIndex, y] + m2Multiplier * m2[xIndex, y];
        }
    }
    static void AddVectorsKernel(Index1D index, VectorView v1, VectorView v2, float v2Multiplier, VectorView result, int stepLength)
    {
        Index1D i, end;
        ComputeIndexes(index, stepLength, result, out i, out end);
        for (; i < end; i++)
        {
            result[i] = v1[i] + v2Multiplier * v2[i];
        }
    }
    static void AddOuterProductKernel(Index1D xIndex, MatrixView matrix, VectorView v1, VectorView v2, float multiplier)
    {
        var ySize = matrix.Extent.Y;
        var v1XValue = v1[xIndex];
        for (int y = 0; y < ySize; y++)
            matrix[xIndex, y] += multiplier * v1XValue * v2[y];
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
    
    static unsafe void MatrixVectorRightSideMulKernel(Index1D index, MatrixView mat, VectorView rightSide, VectorView result)
    {
        float num = 0f;
        fixed(float* rowStart = &mat[index,0])
        fixed(float* vectorStart = &rightSide[0])
        {
            float* rowEnd = rowStart+mat.Extent.Y;
            float* currentVectorPos = vectorStart;
            for (float* current=rowStart; current < rowEnd; current++)
            {
                num += *current * *currentVectorPos;
                currentVectorPos++;
            }
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
    static void MatrixL2Kernel(Index1D index, MatrixView matrix, VectorView result){
        var sum = 0.0f;
        for (int i = 0;i<matrix.Extent.Y;i++){
            float tmp = matrix[index, i];
            sum +=tmp*tmp;
        }
        result[index]=sum;
    }
}