global using RentedArraySharp;
global using System.Collections;
global using GradientDescentSharp;
global using GradientDescentSharp.Utils;
global using GradientDescentSharp.DataAccessors;
global using GradientDescentSharp.GradientDescents;
global using GradientDescentSharp.ComplexDataStructures;
global using GradientDescentSharp.NeuralNetwork.WeightInitializers;
global using GradientDescentSharp.NeuralNetwork.ActivationFunction;
global using Playground;
using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra.Single;
using GradientDescentSharp.Utils.Kernels;
using ILGPU.Runtime;

public static class Test
{
    public static void GpuCpuPerformance()
    {
        var watch = new Stopwatch();
        watch.Start();
        // Examples.NeuralNetworkClassificationExample();

        var rows = 1200;
        var cols = 1500;
        var cols2 = 1000;

        var mat1 = DenseMatrix.Create(rows, cols, (_, _) => Random.Shared.NextSingle());
        var mat2 = DenseMatrix.Create(cols, cols2, (_, _) => Random.Shared.NextSingle());

        LinearAlgebraProvider.CompileAllKernels();
        using var provider = LinearAlgebraProvider.Create();
        using var gmat1 = provider.Provider.CreateMatrix(rows, cols);
        using var gmat2 = provider.Provider.CreateMatrix(cols, cols2);
        using var gmat3 = provider.Provider.CreateMatrix(rows, cols2);

        gmat1.CopyFromCPU(mat1.ToArray());
        gmat2.CopyFromCPU(mat2.ToArray());
        Console.WriteLine("Data creation/compile " + watch.ElapsedMilliseconds);
        watch.Restart();
        var mat3 = mat1 * mat2;
        Console.WriteLine("Cpu mat mul " + watch.ElapsedMilliseconds);

        watch.Restart();
        provider.Provider.MatrixMul(gmat1, gmat2, gmat3);
        gmat3.View.At(0, 0); //evaluate matrix multiplication

        Console.WriteLine("Gpu mat mul " + watch.ElapsedMilliseconds);

    }
}