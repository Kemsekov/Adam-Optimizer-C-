using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AdamOptimizer
{
    public static class MeasurePerformance
    {
        public static void Measure(Func<double[],double> func)
        {
            var variables1 = new double[] { Random.Shared.NextDouble(), Random.Shared.NextDouble(), Random.Shared.NextDouble() };
            var variables2 = new double[3];
            variables1.CopyTo(variables2, 0);

            var gradientDescend1 = new GradientDescend(variables1, func);
            var gradientDescend2 = new GradientDescend(variables2, func);

            var before = func(variables1);

            var maxIterations = 40;
            var learningRate = 0.05;
            var theta = 0.0001;
            int adamCount = 0;
            int mineCount = 0;
            for (int i = 0; i < 1000; i++)
            {
                variables1 = new double[] { Random.Shared.NextDouble(), Random.Shared.NextDouble(), Random.Shared.NextDouble() };
                variables2 = new double[3];
                variables1.CopyTo(variables2,0);

                before = func(variables1);
                gradientDescend1 = new GradientDescend(variables1, func);
                gradientDescend2 = new GradientDescend(variables2, func);
                gradientDescend1.AdamDescent(maxIterations, learningRate, theta);
                gradientDescend2.MineDescent(maxIterations, learningRate, theta);
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