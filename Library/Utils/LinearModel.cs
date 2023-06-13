using System.Diagnostics;
using GradientDescentSharp.Utils.Kernels;
using ILGPU;
using ILGPU.Runtime;
using MathNet.Numerics.LinearAlgebra.Double;

namespace GradientDescentSharp.Utils;

/// <summary>
/// Linear model defined on some basis, that can be used to make predictions. <br/>
/// Call <see cref="MseBestFit"/> to build one.
/// </summary>
public class LinearModel{
    /// <summary>
    /// Finds best MSE fit of parameters for model <br/>
    /// Y=B[0]*A[0](X)+B[1]*A[1](X)...B[n]*A[n](X) <br/>
    /// where X - is a array of regressors, function input vector <br/>
    /// B[i] - is parameters vector(return of this function) <br/>
    /// A[i] - some function over regressors X <br/>
    /// </summary>
    /// <param name="basis">Array of basis functions, for each of this functions following </param>
    /// <param name="inputX">Input vectors, X values</param>
    /// <param name="y">Corresponding to inputX, Y values</param>
    /// <returns>Linear model that can be used to make predictions</returns>
    public static LinearModel MseBestFit(Func<double[], double>[] basis, IEnumerable<double[]> inputX, IEnumerable<double> y)
    {
        var C = DenseMatrix.Create(basis.Length, basis.Length, 0);
        var E = DenseVector.Create(basis.Length, 0);
        var B = DenseVector.Create(basis.Length, 0);

        foreach (var d in inputX.Zip(y))
        {
            var a_i = DenseVector.Create(basis.Length, i => basis[i](d.First)).Values;
            C.MapIndexedInplace((i, j, x) => x + a_i[i] * a_i[j]);
            E.MapIndexedInplace((i, x) => (float)(x + d.Second * a_i[i]));
        }
        
        C.Solve(E, B);
        return new(basis,B);
    }

    public static LinearModel MseBestFitGpu(Func<double[], double>[] basis, IEnumerable<double[]> inputX, IEnumerable<double> y)
    {
        using var context = Context.CreateDefault();
        using var accelerator = context.GetPreferredDevice(false).CreateAccelerator(context);
        using var matrix = accelerator.Allocate2DDenseY<float>((basis.Length,basis.Length));
        using var a_i_Gpu = accelerator.Allocate1D<float>(basis.Length);
        
        var provider = new LinearAlgebraProvider(accelerator);
        var a_i = new float[basis.Length];
        var E = MathNet.Numerics.LinearAlgebra.Single.DenseVector.Create(basis.Length, 0);
        foreach (var d in inputX.Zip(y))
        {
            for(int i = 0;i<basis.Length;i++)
                a_i[i]=(float)basis[i](d.First);
            a_i_Gpu.CopyFromCPU(a_i);
            provider.AddOuterProduct(matrix,a_i_Gpu,a_i_Gpu,1);
            E.MapIndexedInplace((i, x) => (float)(x + d.Second * a_i[i]));
        }

        var B = MathNet.Numerics.LinearAlgebra.Single.DenseVector.Create(basis.Length, 0);
        var C = MathNet.Numerics.LinearAlgebra.Single.DenseMatrix.Create(basis.Length,basis.Length,0);
        matrix.View.BaseView.AsGeneral().CopyToCPU(C.Values);
        C.Solve(E, B);
        return new(basis,DenseVector.Create(basis.Length,i=>B[i]));
    }

    public LinearModel(Func<double[], double>[] basis, DenseVector parameters){
        Basis = basis;
        Parameters = parameters;
    }
    /// <summary>
    /// Predicts Y value of input vector
    /// </summary>
    public double Predict(double[] input){
        double result = 0;
        for(int i = 0;i<Basis.Length;i++){
            result+=Basis[i](input)*Parameters[i];
        }
        return result;
    }
    public Func<double[], double>[] Basis { get; }
    public DenseVector Parameters { get; }
}
