using MathNet.Numerics;
using MathNet.Numerics.Integration;
using MathNet.Numerics.Statistics;

namespace Playground;
public partial class Examples
{
    public static void SampleDistribution()
    {
        double f_x(double x, double y){
            if(x<0 || y<0) return 0;
            var up = Math.Sqrt(x)+2*y;
            return 0.119918069*up*up/(10+x+y+Math.Exp(x+y));
        }
        double F_x(double x, double y){
            return
            Integrate.GaussLegendre(x1=>
            Integrate.GaussLegendre(y1=>
                f_x(x1,y1),
            0.000001,y),
            0.000001,x);
        }
        double f_z(double z){
            return 
            Integrate.GaussLegendre(x=>
                f_x(x,(z-Math.Log(x+1))/x)/x,
            0.000001,100);
        }
        double xUpperBound = 10;
        double yUpperBound = 8;
        double[] Sample(){
            var variables = new ArrayDataAccess<double>(2);
            variables[0] = Random.Shared.NextDouble()*xUpperBound;
            variables[1] = Random.Shared.NextDouble()*yUpperBound;
            var p = Random.Shared.NextDouble();
            var descent = new MineDescent(variables,vars=>Math.Abs(F_x(vars[0],vars[1])-p)){
                // Logger=new ConsoleLogger(),
                DescentRate=1
            };
            descent.Descent(40);
            return variables.Array;
        }
        var samples = Enumerable.Range(0,1000).ToArray().AsParallel().Select(t=>Sample()).Select(t=>t[0]*t[1]+Math.Log(1+t[0])).ToArray();

        System.Console.WriteLine("Mean\t"+Statistics.Mean(samples));
        System.Console.WriteLine("Stddev\t"+Statistics.StandardDeviation(samples));
        System.Console.WriteLine("X_med\t"+Statistics.Median(samples));
        System.Console.WriteLine("Skewness\t"+Statistics.Skewness(samples));
        System.Console.WriteLine("80% percentile\t"+Statistics.Percentile(samples,80));
        
    }
}