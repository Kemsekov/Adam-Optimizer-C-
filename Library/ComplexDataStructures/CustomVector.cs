using MathNet.Numerics.LinearAlgebra.Storage;
namespace GradientDescentSharp.ComplexDataStructures;
/// <summary>
/// Custom vector that store it's values on data accessors that can be used by gradient descent
/// </summary>
public class CustomVector : Vector
{
    ///<inheritdoc/>
    public CustomVector(VectorStorage<FloatType> storage) : base(storage)
    {
    }
}
class CustomVectorStorage : VectorStorage<FloatType>
{
    public override bool IsDense => true;

    public int StartIndex { get; }
    public IDataAccess<FloatType> Data { get; }

    public CustomVectorStorage(IDataAccess<FloatType> data, int startIndex, int length) : base(length)
    {
        StartIndex = startIndex;
        Data = data;
    }
    public override FloatType At(int index)
    {
        return Data[StartIndex + index];
    }

    public override void At(int index, FloatType value)
    {
        Data[StartIndex + index] = value;
    }
}