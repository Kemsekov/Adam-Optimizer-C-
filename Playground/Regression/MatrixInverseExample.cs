using MathNet.Numerics.LinearAlgebra.Double;

namespace Playground;
public partial class Examples
{
    public static void LinearRegression()
    {
        var A = new Func<double[], double>[]{
            x=>1,
            x=>Math.Pow(x[0],1),
            x=>Math.Pow(x[0],2),
            x=>Math.Pow(x[0],3),
            x=>Math.Pow(x[0],4),
        };

        var data = Enumerable.Range(0, 10).Select(i =>
        {
            var x = (i - 5.0) / 2;
            return new[] { x };
        }).ToArray();
        var y = data.Select(m =>
        {
            var x = m[0];
            var x2 = x * x;
            var x3 = x * x * x;
            var x4 = x * x * x * x;
            return x2 + x3 / 10 - x4 / 5 + 2;
        });

        var model = LinearModel.MseBestFit(A, data, y);
        System.Console.WriteLine("Parameters for y=x^2+x^3/10-x^4/5+2");
        PrintVector(model.Parameters);


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