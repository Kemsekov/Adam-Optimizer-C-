namespace Playground;
public partial class Examples
{
    static (int adam, int mine) MeasureDescentsPerformance(Func<IDataAccess<double>, double> func, int variablesLength)
    {
        ArrayDataAccess<double> variables1 = new double[] { Random.Shared.NextDouble(), Random.Shared.NextDouble(), Random.Shared.NextDouble() };
        ArrayDataAccess<double> variables2 = new double[variablesLength];

        for (int i = 0; i < variablesLength; i++) variables1[i] = Random.Shared.NextDouble() * 2 - 1;
        variables1.Array.CopyTo(variables2, 0);

        var adamDescent = new AdamDescent(variables1, func);
        var mineDescent = new MineDescent(variables2, func);

        double before;

        var maxIterations = 40;
        var descentRate = 0.05;
        var theta = 0.0001;
        int adamCount = 0;
        int mineCount = 0;

        for (int i = 0; i < 1000; i++)
        {
            variables1 = new double[3];
            variables2 = new double[3];

            for (int k = 0; k < variables1.Length; k++)
                variables1[k] = Random.Shared.NextDouble() * 2 - 1;

            variables1.Array.CopyTo(variables2, 0);

            before = func(variables1);
            adamDescent = new AdamDescent(variables1, func)
            {
                Theta = theta,
                DescentRate = descentRate
            };
            mineDescent = new MineDescent(variables2, func)
            {
                Theta = theta,
                DescentRate = descentRate
            };

            adamDescent.Descent(maxIterations);
            mineDescent.Descent(maxIterations);
            var adam = func(variables1);
            var mine = func(variables2);
            if (adam <= mine)
                adamCount++;
            else
                mineCount++;
        }
        System.Console.WriteLine("Adam : " + adamCount);
        System.Console.WriteLine("Mine : " + mineCount);
        System.Console.WriteLine("----------------------");
        return (adamCount, mineCount);
    }
    public static void MeasureDescentsPerformanceAvg()
    {
        var results = Functions.Select(x => MeasureDescentsPerformance(x, 3)).ToArray();
        var (adam, mine) = results.Aggregate((x1, x2) => (x1.adam + x2.adam, x1.mine + x2.mine));
        mine /= results.Length;
        adam /= results.Length;
        System.Console.WriteLine("Average adam " + adam);
        System.Console.WriteLine("Average mine " + mine);
    }
}