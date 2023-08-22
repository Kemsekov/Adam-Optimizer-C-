namespace GradientDescentSharp.LinearAlgebra;
//TODO: COMPLETE
//this is not full and proper definition of vector and matrix,
//but rather it is a proxy interface to any proper implementation
//definition here is just a bone minimum that I currently use for 
//my neural network models

public abstract class Vector : IEnumerable<float>{
    public abstract float this[int index]{get;set;}
    public int Count{get;}
    public Vector(int length){
        Count = length;
    }
    public abstract void Add(Vector vector, Vector result, float multiplier);
    public abstract IEnumerator<float> GetEnumerator();
    public abstract Vector MapIndexed(Func<int, float, float> f);
    public Vector Map(Func<float, float> f)=>MapIndexed((index,x)=>f(x));
    public abstract void MapIndexedInplace(Func<int, float, float> f);
    public abstract float L2Norm();
    /// <summary>
    /// Creates a vector of same type with given length
    /// </summary>
    public abstract Vector Create(int length);
    public static Vector operator+(Vector v1, Vector v2){
        var v3 = v1.Create(v1.Count);
        v1.Add(v2,v3,1);
        return v3;
    }
    public static Vector operator-(Vector v1, Vector v2){
        var v3 = v1.Create(v1.Count);
        v1.Add(v2,v3,-1);
        return v3;
    }
    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}
public abstract class Matrix
{
    public int RowCount{get;}
    public int ColumnCount{get;}
    public abstract float this[int row,int column]{get;set;}
    public Matrix(int rowCount, int columnCount){
        RowCount = rowCount;
        ColumnCount = columnCount;
    }
    public void MapInplace(Func<float, float> f)=>MapIndexedInplace((x,y,v)=>f(v));
    public abstract void MultiplyLeft(Vector vector,Vector result);
    public abstract void MultiplyRight(Vector vector,Vector result);
    public abstract void AddOuterProduct(Vector v1,Vector v2, float multiplier);
    public abstract void MapIndexedInplace(Func<int,int,float, float> f);
    public static Vector operator*(Matrix mat, Vector vec){
        var result = vec.Create(mat.RowCount);
        mat.MultiplyRight(vec,result);
        return result;
    }
    public static Vector operator*(Vector vec,Matrix mat){
        var result = vec.Create(mat.ColumnCount);
        mat.MultiplyLeft(vec,result);
        return result;
    }
}