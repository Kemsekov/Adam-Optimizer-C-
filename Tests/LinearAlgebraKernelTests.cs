using GradientDescentSharp.Utils.Kernels;
using ILGPU;
using ILGPU.Runtime;
using MathNet.Numerics.LinearAlgebra.Single;

namespace Tests;
public class GpuContextFixture : IDisposable
{
    public GpuContextFixture()
    {
        Context = Context.Create(builder =>
        {
            builder.Optimize(OptimizationLevel.O2);
        });
        Accelerator = Context.GetPreferredDevice(false).CreateAccelerator(Context);
        Provider = new LinearAlgebraProvider(Accelerator);
    }

    public Context Context { get; }
    public Accelerator Accelerator { get; }
    public LinearAlgebraProvider Provider { get; }

    public void Dispose()
    {
        Accelerator.Dispose();
        Context.Dispose();
    }
}

public class LinearAlgebraKernelTests : IClassFixture<GpuContextFixture>
{
    public LinearAlgebraKernelTests(GpuContextFixture context)
    {
        Context = context;
    }
    public GpuContextFixture Context { get; }

    [Fact]
    public void AddMatricies()
    {
        for (int k = 0; k < 10; k++)
        {
            var rows = Random.Shared.Next(10) + 1;
            var cols = Random.Shared.Next(10) + 1;

            using var gpuMat1 = Context.Accelerator.Allocate2DDenseY<float>((rows, cols));
            using var gpuMat2 = Context.Accelerator.Allocate2DDenseY<float>((rows, cols));
            using var gpuMat3 = Context.Accelerator.Allocate2DDenseY<float>((rows, cols));

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    gpuMat1.View.At(i, j, Random.Shared.NextSingle());
                    gpuMat2.View.At(i, j, Random.Shared.NextSingle());
                }
            }

            var mul = Random.Shared.NextSingle() * 2 - 1;
            Context.Provider.AddMatricies((rows, cols), gpuMat1, gpuMat2, mul, gpuMat3);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    var v1 = gpuMat1.View.At(i,j);
                    var v2 = gpuMat2.View.At(i,j);
                    var v3 = v1 + mul * v2;
                    Assert.Equal(gpuMat3.View.At(i,j), v3);
                }
            }
        }
    }

    [Fact]
    public void AddVectors()
    {
        for (int k = 0; k < 10; k++)
        {
            var rows = Random.Shared.Next(10) + 1;
            using var gpuVec1 = Context.Accelerator.Allocate1D<float>(rows);
            using var gpuVec2 = Context.Accelerator.Allocate1D<float>(rows);
            using var gpuVec3 = Context.Accelerator.Allocate1D<float>(rows);

            for (int i = 0; i < rows; i++)
            {
                gpuVec1.View.At(i, Random.Shared.NextSingle());
                gpuVec2.View.At(i, Random.Shared.NextSingle());
            }
            var mul = Random.Shared.NextSingle() * 2 - 1;
            Context.Provider.AddVectors(rows, gpuVec1, gpuVec2, mul, gpuVec3);

            for (int i = 0; i < rows; i++)
            {
                var v1 = gpuVec1.View.At(i);
                var v2 = gpuVec2.View.At(i);
                var v3 = v1 + mul * v2;
                Assert.Equal(gpuVec3.View.At(i), v3);
            }
        }
    }
    [Fact]
    public void MultiplyMatrixVector()
    {
        for (int k = 0; k < 10; k++)
        {
            var rows = Random.Shared.Next(10) + 1;
            var cols = Random.Shared.Next(10) + 1;

            var vec1 = DenseVector.Create(rows, x => Random.Shared.NextSingle());
            var vec2 = DenseVector.Create(cols, x => Random.Shared.NextSingle());
            var mat = DenseMatrix.Create(rows, cols, (i, j) => Random.Shared.NextSingle());

            using var gpuMat = Context.Accelerator.Allocate2DDenseY<float>((rows, cols));
            using var gpuVec1 = Context.Accelerator.Allocate1D<float>(rows);
            using var gpuVec2 = Context.Accelerator.Allocate1D<float>(cols);
            gpuMat.CopyFromCPU(mat.ToArray());
            gpuVec1.CopyFromCPU(vec1.Values);
            gpuVec2.CopyFromCPU(vec2.Values);

            var expectedLeft = vec1 * mat;
            var expectedRight = mat * vec2;

            using var actualLeft = Context.Accelerator.Allocate1D<float>(expectedLeft.Count);
            using var actualRight = Context.Accelerator.Allocate1D<float>(expectedRight.Count);

            Context.Provider.MatrixVectorLeftSideMul((int)actualLeft.Extent, gpuMat, gpuVec1, actualLeft);
            Context.Provider.MatrixVectorRightSideMul((int)actualRight.Extent, gpuMat, gpuVec2, actualRight);

            for (int i = 0; i < expectedRight.Count; i++)
            {
                Assert.Equal(expectedRight[i], actualRight.View.At(i));
            }
            for (int i = 0; i < expectedLeft.Count; i++)
            {
                Assert.Equal(expectedLeft[i], actualLeft.View.At(i));
            }
        }
    }

    [Fact]
    public void MultiplyMatrices()
    {
        for (int k = 0; k < 10; k++)
        {
            var rows = Random.Shared.Next(10) + 1;
            var cols = Random.Shared.Next(10) + 1;
            var cols2 = Random.Shared.Next(10) + 1;
            var mat1 = DenseMatrix.Create(rows, cols, (i, j) => Random.Shared.NextSingle());
            var mat2 = DenseMatrix.Create(cols, cols2, (i, j) => Random.Shared.NextSingle());
            var expected = mat1 * mat2;

            using var gpuMat1 = Context.Accelerator.Allocate2DDenseY<float>((rows, cols));
            using var gpuMat2 = Context.Accelerator.Allocate2DDenseY<float>((cols, cols2));
            using var actual = Context.Accelerator.Allocate2DDenseY<float>((rows, cols2));

            gpuMat1.CopyFromCPU(mat1.ToArray());
            gpuMat2.CopyFromCPU(mat2.ToArray());
            Context.Provider.MatrixMul(((Index2D)actual.Extent), gpuMat1, gpuMat2, actual);
            for (int i = 0; i < expected.RowCount; i++)
                for (int j = 0; j < expected.ColumnCount; j++)
                {
                    var difference = Math.Abs(expected[i, j] - actual.View.At(i, j));
                    Assert.True(difference < 0.0001);
                }
        }
    }

    [Fact]
    public void AddOuterProductKernel()
    {
        for (int k = 0; k < 10; k++)
        {
            var rows = Random.Shared.Next(10) + 1;
            var cols = Random.Shared.Next(10) + 1;
            var mat = DenseMatrix.Create(rows, cols, (i, j) => Random.Shared.NextSingle());
            var vec1 = DenseVector.Create(rows, x => Random.Shared.NextSingle());
            var vec2 = DenseVector.Create(cols, x => Random.Shared.NextSingle());

            using var gpuMat = Context.Accelerator.Allocate2DDenseY<float>((rows, cols));
            using var gpuVec1 = Context.Accelerator.Allocate1D<float>(rows);
            using var gpuVec2 = Context.Accelerator.Allocate1D<float>(cols);
            gpuMat.CopyFromCPU(mat.ToArray());
            gpuVec1.CopyFromCPU(vec1.Values);
            gpuVec2.CopyFromCPU(vec2.Values);

            //check that matrices and vectors are copied
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                {
                    Assert.Equal(mat[i, j], gpuMat.View.At(i, j));
                }
            for (int i = 0; i < rows; i++)
                Assert.Equal(vec1[i], gpuVec1.View.At(i));
            for (int i = 0; i < cols; i++)
                Assert.Equal(vec2[i], gpuVec2.View.At(i));

            var multiplier = Random.Shared.NextSingle() * 2 - 1; //[-1;1]
            var expected = mat + multiplier * vec1.OuterProduct(vec2);

            Context.Provider.AddOuterProduct((rows, cols), gpuMat, gpuVec1, gpuVec2, multiplier);

            //because float32 on cpu and gpu works differently, it is not guaranteed that
            //we will get same results, so the only way to check correctness is 
            //to ensure that absolute difference is small.
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                {
                    var difference = Math.Abs(expected[i, j] - gpuMat.View.At(i, j));
                    Assert.True(difference < 0.0001);
                }

        }
    }
}