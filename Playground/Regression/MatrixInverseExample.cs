using MathNet.Numerics.LinearAlgebra.Double;

namespace Playground;
public partial class Examples
{
    public static void LinearRegression()
    {
        var A = new Func<double[], double>[50];
        for(int i = 0;i<A.Length;i++){
            var iLocal = i;
            A[i]=x=>Math.Pow(x[0],iLocal);
        }
        var rand = new Random(0);
        var data = Enumerable.Range(0, 5000).Select(i =>
        {
            return new[] { rand.NextDouble()*4-2 };
        }).ToArray();
        var y = data.Select(m =>
        {
            var x = m[0];
            var x2 = x * x;
            var x3 = x * x * x;
            var x4 = x * x * x * x;
            return x2 + x3 / 10 - x4 / 5 + 2+0.1*(rand.NextDouble()-0.5);
        });

        var model = LinearModel.MseBestFit(A, data, y);
        System.Console.WriteLine("Parameters for y=x^2+x^3/10-x^4/5+2");
        // PrintVector(model.Parameters);


        static void PrintVector(DenseVector B)
        {
            System.Console.Write('[');
            foreach (var value in B.SkipLast(1))
                System.Console.Write(value.ToString("0.0000") + ",");
            System.Console.Write(B.Last().ToString("0.0000"));
            System.Console.WriteLine("]");
        }
    }
}