using MathNet.Numerics.LinearAlgebra.Double;

namespace GradientDescentSharp.Utils;
public static class MathUtils
{
    /// <summary>
    /// Moves data points from each other to normally fill input vector space
    /// </summary>
    public static void DistributeData(Vector[] data, double CoordinatesScale, Vector CoordinatesShift)
    {
        var InputVectorLength = CoordinatesShift.Count;
        void normalizeVector(ref DenseVector position, double step)
        {
            for (int k = 0; k < InputVectorLength; k++)
            {
                if (position[k] >= 1 - step / 2)
                {
                    position[k] = 0;
                    if (k + 1 >= InputVectorLength) break;
                    position[k + 1] += step;
                }
            }
        }
        var chunkSize = MathF.Pow(data.Length, 1f / InputVectorLength);
        var step = 1f / chunkSize;
        var position = new DenseVector(new double[InputVectorLength]);

        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (Vector)position.Clone();
            position[0] += step;
            normalizeVector(ref position, step);
        }
        NormalizeCoordinates(data);
        foreach(var d in data)
        {
            d.MapIndexedInplace((index,value)=>value*CoordinatesScale+CoordinatesShift[index]);
        }
    }
    /// <summary>
    /// Makes sure that input(non missing) part of data is filling bounds
    /// in range [0,1]. 
    /// Like apply linear transformation to input part of vectors in data that it
    /// fills [0,1] space.
    /// </summary>
    public static void NormalizeCoordinates(Vector[] data, Vector? input = null)
    {
        if(data.Length==0) return;
        var InputVectorLength = data[0].Count;
        input ??= new DenseVector(new double[InputVectorLength]);
        var maxArray = new double[InputVectorLength];
        var minArray = new double[InputVectorLength];;
        Array.Fill(maxArray,double.MinValue);
        Array.Fill(minArray,double.MaxValue);
        Vector max = new DenseVector(maxArray);
        Vector min = new DenseVector(minArray);

        for (int i = 0; i < data.Length; i++)
        {
            var dt = data[i];
            max = (Vector)(max.PointwiseMaximum(dt));
            min = (Vector)(min.PointwiseMinimum(dt));
        }
        var diff = max-min;
        for (int i = 0; i < data.Length; i++)
        {
            var dt = data[i];

            dt.MapIndexedInplace((index,x)=>{
                if(input[index]<-1) return x;
                return (x-min[index])/(diff[index]);
            });
        }
    }
}