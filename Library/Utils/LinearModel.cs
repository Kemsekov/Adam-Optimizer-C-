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
            var a_i = DenseVector.Create(basis.Length, i => basis[i](d.First));
            C.MapIndexedInplace((i, j, x) => x + a_i[i] * a_i[j]);
            E.MapIndexedInplace((i, x) => (float)(x + d.Second * a_i[i]));
        }

        C.Solve(E, B);
        return new(basis,B);
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
