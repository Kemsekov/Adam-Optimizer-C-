using MathNet.Numerics.LinearAlgebra.Storage;
namespace GradientDescentSharp.ComplexDataStructures;
/// <summary>
/// Custom vector that store it's values on data accessors that can be used by gradient descent
/// </summary>
public class CustomVector : Vector
{
    ///<inheritdoc/>
    public CustomVector(VectorStorage<double> storage) : base(storage)
    {
    }
}
class CustomVectorStorage : VectorStorage<double>
{
    public override bool IsDense => true;

    public int StartIndex { get; }
    public IDataAccess<double> Data { get; }

    public CustomVectorStorage(IDataAccess<double> data, int startIndex, int length) : base(length)
    {
        StartIndex = startIndex;
        Data = data;
    }
    public override double At(int index)
    {
        return Data[StartIndex + index];
    }

    public override void At(int index, double value)
    {
        Data[StartIndex + index] = value;
    }
}