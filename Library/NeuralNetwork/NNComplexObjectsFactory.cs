using MathNet.Numerics.LinearAlgebra.Single;
namespace GradientDescentSharp.NeuralNetwork;
///<inheritdoc/>
public class NNComplexObjectsFactory : IComplexObjectsFactory<float>
{
    ///<inheritdoc/>
    public FMatrix CreateMatrix(int rows, int columns)
        => DenseMatrix.Create(rows, columns, 0);

    ///<inheritdoc/>
    public FVector CreateVector(int length)
        => DenseVector.Create(length, 0);
}
