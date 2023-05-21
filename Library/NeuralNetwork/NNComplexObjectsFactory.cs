using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork;

public class NNComplexObjectsFactory : IComplexObjectsFactory<float>{
    public Matrix<float> CreateMatrix(int rows, int columns)
        => DenseMatrix.Create(rows,columns,0);

    public Vector<float> CreateVector(int length)
        => DenseVector.Create(length,0);
}
