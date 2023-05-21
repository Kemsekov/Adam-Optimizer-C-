using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork;

///<inheritdoc/>
public class NNComplexObjectsFactory : IComplexObjectsFactory<float>
{
    ///<inheritdoc/>
    public Matrix<float> CreateMatrix(int rows, int columns)
        => NNMatrix.Create(rows, columns, 0);

    ///<inheritdoc/>
    public Vector<float> CreateVector(int length)
        => DenseVector.Create(length, 0);
}
