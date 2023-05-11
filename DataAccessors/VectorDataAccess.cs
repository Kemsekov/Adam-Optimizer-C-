namespace GradientDescentSharp.DataAccessors;
public class VectorDataAccess : IDataAccess<double>
{
    public static VectorDataAccess Of(Vector vector)
    {
        return new(vector);
    }
    public VectorDataAccess(Vector vector)
    {
        Vector = vector;
    }
    public VectorDataAccess(int length)
    {
        Vector = DenseVector.Create(length, 0);
    }
    public double this[int index]
    {
        get => Vector[index];
        set => Vector[index] = value;
    }

    public Vector Vector { get; }

    public int Length => Vector.Count;
    public IEnumerator<double> GetEnumerator()
    {
        foreach (var v in Vector) yield return v;
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

    public static implicit operator Vector(VectorDataAccess t) => t.Vector;
    public static implicit operator VectorDataAccess(Vector t) => new(t);
}