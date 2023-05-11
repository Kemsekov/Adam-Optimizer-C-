namespace GradientDescentSharp;
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
    public static void FindMatrixInverse()
    {
        var mat = new DenseMatrix(2, 2);
        mat[0, 0] = 1;
        mat[0, 1] = 2;
        mat[1, 0] = 3;
        mat[1, 1] = 4;
        var configuration = new BestSolutionConfiguration(
            variablesLength: 8
        );
        var result = TryToFindBestSolution(x =>
        {
            var factory = new ComplexObjectsFactory(x);
            var m1 = factory.TakeMatrix(2, 2);
            var m2 = factory.TakeMatrix(2, 2);
            var m3 = m1 * m2;
            return Math.Abs(m3[0, 0] - 1) + Math.Abs(m3[0, 1]) + Math.Abs(m3[1, 0]) + Math.Abs(m3[1, 1] - 1);
        }, configuration);

        var factory = new ComplexObjectsFactory(result);
        var m1 = factory.TakeMatrix(2, 2);
        var m2 = factory.TakeMatrix(2, 2);
        var m3 = m1 * m2;
        Console.WriteLine(m1);
        Console.WriteLine(m2);
        Console.WriteLine("Their product is");
        Console.WriteLine(m3);
    }
    public record BestSolutionConfiguration(int variablesLength, int solutionsCount = 100, Action<IDataAccess<double>>? init = null, int descentIterations = 100, double descentRate = 0.1, double theta = 0.0001);
    public static IDataAccess<double> TryToFindBestSolution(Func<IDataAccess<double>, double> func, BestSolutionConfiguration configuration)
    {
        IDataAccess<double> variables2 = new ArrayDataAccess<double>(configuration.variablesLength);
        if (configuration.init is null)
            for (int i = 0; i < configuration.variablesLength; i++)
                variables2[i] = Random.Shared.NextDouble() * 4 - 2;
        configuration.init?.Invoke(variables2);

        ArrayDataAccess<double> bestMine = (variables2.ToArray());

        var descentIterations = configuration.descentIterations;
        var descentRate = configuration.descentRate;
        var theta = configuration.theta;
        var before = func(variables2);
        Console.WriteLine("Starts at " + before);

        for (int k = 0; k < configuration.solutionsCount; k++)
        {
            if (configuration.init is null)
                for (int i = 0; i < configuration.variablesLength; i++)
                    variables2[i] = Random.Shared.NextDouble() * 4 - 2;
            configuration.init?.Invoke(variables2);

            var mineDescent = new MineDescent(variables2, func)
            {
                DescentRate = descentRate,
                Theta = theta
            };

            System.Console.WriteLine("Before " + func(variables2));

            mineDescent.Descent(descentIterations);
            var currentScore = func(variables2);
            var bestScore = func(bestMine);
            System.Console.WriteLine("After " + currentScore);
            System.Console.WriteLine("-----------");
            if (currentScore < bestScore)
            {
                bestMine = variables2.ToArray();
            }
        }
        var mineScore = func(bestMine);
        Console.WriteLine("Mine " + mineScore);
        System.Console.WriteLine("Mine Point is [" + String.Join(' ', variables2.Select(x => x.ToString("0.000"))) + "]");
        return bestMine;
    }
    public static void PrintBothAdamAndMinePerformance(Func<IDataAccess<double>, double> func, int variablesLength, Action<IDataAccess<double>>? init = null)
    {
        //choose function to find it's minima
        // var func = (double a, double b, double c)=>Math.Abs(Functions.Sum(x=>x(a,b,c)));

        ArrayDataAccess<double> variables1 = new double[variablesLength];
        ArrayDataAccess<double> variables2 = new double[variablesLength];

        if (init is null)
            for (int i = 0; i < variablesLength; i++)
                variables1[i] = Random.Shared.NextDouble() * 2 - 1;
        init?.Invoke(variables1);

        variables1.Array.CopyTo(variables2, 0);
        var maxIterations = 500;
        var descentRate = 1;
        var theta = 0.0001;

        var adamDescent = new AdamDescent(variables1, func)
        {
            Theta = theta,
            DescentRate = descentRate
        };
        var mineDescent = new MineDescent(variables2, func)
        {
            Theta = theta,
            DescentRate = descentRate
        };

        var before = func(variables1);

        System.Console.WriteLine(adamDescent.Descent(maxIterations) + " Adam iterations");
        System.Console.WriteLine(mineDescent.Descent(maxIterations) + " Mine iterations");

        var after1 = func(variables1);
        var after2 = func(variables2);

        Console.WriteLine("Before " + before);
        Console.WriteLine("Adam " + after1);
        Console.WriteLine("Mine " + after2);
        System.Console.WriteLine("Adam Point is [" + String.Join(' ', variables1.Select(x => x.ToString("0.000"))) + "]");
        System.Console.WriteLine("Mine Point is [" + String.Join(' ', variables2.Select(x => x.ToString("0.000"))) + "]");
    }
    public static void MeasureAvg()
    {
        var results = Functions.Select(x => Measure(x, 3)).ToArray();
        var (adam, mine) = results.Aggregate((x1, x2) => (x1.adam + x2.adam, x1.mine + x2.mine));
        mine /= results.Length;
        adam /= results.Length;
        System.Console.WriteLine("Average adam " + adam);
        System.Console.WriteLine("Average mine " + mine);
    }
    public static (int adam, int mine) Measure(Func<IDataAccess<double>, double> func, int variablesLength)
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
    record Model(CustomMatrix[] layers, IDataAccess<double> freeCoefficients);
    public static void TryingML()
    {
        MathNet.Numerics.LinearAlgebra.Vector<double> ActivationFunction(MathNet.Numerics.LinearAlgebra.Vector<double> input)
        {
            input.MapInplace(x => 1.0 / (1 + Math.Exp(-x)));
            return input;
        }
        Model BuildModel(IDataAccess<double> x)
        {
            //outputSize1,inputSize
            //outputSize2, outputSize1
            //and so on

            var factory = new ComplexObjectsFactory(x);
            var layers = new[]{
                    factory.TakeMatrix(5, 3),
                    factory.TakeMatrix(6, 5),
                    factory.TakeMatrix(3, 6),
                    factory.TakeMatrix(1, 3)
                };
            var freeCoefficients = factory.TakeDouble(layers.Length);
            return new(layers, freeCoefficients);
        }
        double Compute(IDataAccess<double> x, Vector input)
        {
            System.Console.WriteLine("-----------");
            var model = BuildModel(x);
            var freeCoefficients = model.freeCoefficients;
            System.Console.WriteLine(input);
            for (int i = 0; i < model.layers.Length; i++)
            {
                var layer = model.layers[i];
                input = (Vector)(ActivationFunction(layer * input) + freeCoefficients[i]);
                System.Console.WriteLine(input);
            }
            return input[0];
        }
        var functionToImitate = Functions[1];
        var inputs = Enumerable.Range(0, 9 * 27).Select(x => DenseVector.Create(3, 0)).ToArray();
        MathUtils.DistributeData(inputs, 2, DenseVector.Create(3, -1));
        var func = (IDataAccess<double> x) =>
        {
            var error = 0.0;
            for (int i = 0; i < inputs.Length; i++)
            {
                var input = inputs[i];
                var expected = functionToImitate(VectorDataAccess.Of(input));
                var result = Compute(x, input);
                error += Math.Pow(result - expected, 2);
            }
            return Math.Sqrt(error);
        };
        void initWeights(IDataAccess<double> x)
        {
            var model = BuildModel(x);
            for (int i = 0; i < model.freeCoefficients.Length; i++)
            {
                model.freeCoefficients[i] = Random.Shared.NextDouble() * 2 - 1;
            }
            foreach (var layer in model.layers)
            {
                layer.MapInplace(x => Random.Shared.NextDouble() * 2 - 1);
                var elementsSum = layer.ReduceColumns((a, b) => a + b).Sum();
                layer.MapInplace(x => x / elementsSum);
            }
        }
        var configuration = new BestSolutionConfiguration(
            variablesLength: 70,
            solutionsCount: 0,
            init: initWeights,
            descentIterations: 100,
            descentRate: 0.5,
            theta: 0.001
        );
        var solution = TryToFindBestSolution(func, configuration);
        System.Console.WriteLine("Running tests");
        var errorSum = 0.0;
        for (int i = 0; i < 20; i++)
        {
            VectorDataAccess input = DenseVector.Create(3, x => Random.Shared.NextDouble());
            var expected = functionToImitate(input);
            var actual = Compute(solution, input);
            var diff = Math.Abs(expected - actual);
            errorSum += diff;
            System.Console.WriteLine($"Expected: {expected.ToString("0.00")}, Actual: {actual.ToString("0.00")}");
            // System.Console.WriteLine("Diff " + diff);
        }
        System.Console.WriteLine("Average error " + errorSum / 20);
        System.Console.WriteLine("-----------");
    }
}