namespace GradientDescentSharp.NeuralNetwork;

public class NNComplexObjectsFactory : IComplexObjectsFactory{
    public Matrix CreateMatrix(int rows, int columns)
        => DenseMatrix.Create(rows,columns,0);

    public Vector CreateVector(int length)
        => DenseVector.Create(length,0);
}
