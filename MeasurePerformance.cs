using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AdamOptimizer
{
    public static class MeasurePerformance
    {
        public static void Measure(Func<IDataAccess<double>, double> func)
        {
            ArrayDataAccess<double> variables1 = new(new double[] { Random.Shared.NextDouble(), Random.Shared.NextDouble(), Random.Shared.NextDouble() });
            ArrayDataAccess<double> variables2 = new(3);
            variables1.Array.CopyTo(variables2.Array, 0);

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
                variables1 = new(new double[] { Random.Shared.NextDouble(), Random.Shared.NextDouble(), Random.Shared.NextDouble() });
                variables2 = new(new double[3]);
                variables1.Array.CopyTo(variables2.Array, 0);

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