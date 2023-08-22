using MathNet.Numerics.LinearAlgebra.Storage;
using MathNet.Numerics.LinearAlgebra.Double;

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

/// <summary>
/// Custom vector that store it's values on data accessors that can be used by gradient descent
/// </summary>
public class CustomVectorSingle : MathNet.Numerics.LinearAlgebra.Single.Vector
{
    ///<inheritdoc/>
    public CustomVectorSingle(VectorStorage<float> storage) : base(storage)
    {
    }
}
