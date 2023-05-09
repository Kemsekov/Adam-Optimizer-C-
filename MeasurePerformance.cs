using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AdamOptimizer
{
    public static class MeasurePerformance
    {
        public static Func<IDataAccess<double>, double>[] Functions ={
                (d)=>Math.Abs(d[0]-d[1]*d[2]+Math.Exp(d[0]*d[1]*d[2])-Math.Sin(d[0]+d[1]+d[2])),
                (d)=>Math.Abs(d[0]-d[1]*d[2]),
                (d)=>Math.Abs(d[0]*d[1]*d[2]-d[0]*Math.Sin(d[2])+Math.Cos(d[0]*d[2])+d[2]),
                (d)=>Math.Abs(d[0]*Math.Cosh(d[0]*d[1]+Math.Sin(d[2]))),
                (d)=>Math.Abs(Math.Tan(d[0]*d[1]+d[2])-Math.Sin(d[1]*d[2]+d[0])),
                (d)=>Math.Abs(Math.Pow(d[0]*d[0],d[1])-d[2]*Math.Exp(d[0])*Math.Log(d[2]*d[2]))
            };
        public static void PrintBothAdamAndMinePerformance(Func<IDataAccess<double>, double> func)
        {
            //choose function to find it's minima
            // var func = (double a, double b, double c)=>Math.Abs(Functions.Sum(x=>x(a,b,c)));

            ArrayDataAccess<double> variables1 = new double[] { Random.Shared.NextDouble(), Random.Shared.NextDouble(), Random.Shared.NextDouble() };
            ArrayDataAccess<double> variables2 = new double[3];
            variables1.Array.CopyTo(variables2, 0);

            var gradientDescent1 = new GradientDescent(variables1, func);
            var gradientDescent2 = new GradientDescent(variables2, func);

            var before = func(variables1);

            var maxIterations = 40;
            var learningRate = 0.1;
            var theta = 0.0001;

            System.Console.WriteLine(gradientDescent1.AdamDescent(maxIterations, learningRate, theta) + " Adam iterations");
            System.Console.WriteLine(gradientDescent2.MineDescent(maxIterations, learningRate, theta) + " Mine iterations");
            var after1 = func(variables1);
            var after2 = func(variables2);

            Console.WriteLine("Before " + before);
            Console.WriteLine("Adam " + after1);
            Console.WriteLine("Mine " + after2);
            System.Console.WriteLine("Adam Point is [" + String.Join(' ', variables1) + "]");
            System.Console.WriteLine("Mine Point is [" + String.Join(' ', variables2) + "]");

        }
        public static void Measure(Func<IDataAccess<double>, double> func)
        {
            ArrayDataAccess<double> variables1 = new double[] { Random.Shared.NextDouble(), Random.Shared.NextDouble(), Random.Shared.NextDouble() };
            ArrayDataAccess<double> variables2 = new double[3];
            variables1.Array.CopyTo(variables2, 0);

            var gradientDescent1 = new GradientDescent(variables1, func);
            var gradientDescent2 = new GradientDescent(variables2, func);

            double before;

            var maxIterations = 40;
            var learningRate = 0.05;
            var theta = 0.0001;
            int adamCount = 0;
            int mineCount = 0;
            for (int i = 0; i < 1000; i++)
            {
                variables1 = new double[] { Random.Shared.NextDouble(), Random.Shared.NextDouble(), Random.Shared.NextDouble() };
                variables2 = new double[3];
                variables1.Array.CopyTo(variables2, 0);

                before = func(variables1);
                gradientDescent1 = new GradientDescent(variables1, func);
                gradientDescent2 = new GradientDescent(variables2, func);
                gradientDescent1.AdamDescent(maxIterations, learningRate, theta);
                gradientDescent2.MineDescent(maxIterations, learningRate, theta);
                var adam = func(variables1);
                var mine = func(variables2);
                if (adam <= mine)
                    adamCount++;
                else
                    mineCount++;
            }
            System.Console.WriteLine("Adam : " + adamCount);
            System.Console.WriteLine("Mine : " + mineCount);
        }
    }
}