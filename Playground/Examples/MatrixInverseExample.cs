namespace Playground;
public partial class Examples
{
    public static void MatrixInverseExample()
    {
        var dimensions = 3;
        var m2 = DenseMatrix.Create(dimensions, dimensions, (x, y) => Random.Shared.NextDouble() * 2 - 1);
        m2.Inverse();
        var identity = DenseMatrix.CreateIdentity(dimensions);
        var errorFunction = (IDataAccess<double> x) =>
        {
            var factory = new ComplexObjectsFactory(x);
            var m1 = factory.CreateMatrix(dimensions, dimensions);
            var m3 = m1 * m2;
            var diff = identity - m3;
            var error = 0.0;
            for (int i = 0; i < dimensions; i++)
                for (int j = 0; j < dimensions; j++)
                {
                    error += Math.Pow(diff[i, j], 2);
                }
            return error;
        };

        var init = (IDataAccess<double> x) =>
        {
            for (int i = 0; i < x.Length; i++)
                x[i] = Random.Shared.NextDouble() * 6 - 3;
        };

        var finder = new BestSolutionFinder(
            variablesLength: dimensions * dimensions,
            data => new MineDescent(data, errorFunction) { Theta = 0.000001, DescentRate = 1 }
        )
        {
            SolutionsCount = 100,
            DescentIterations = 20,
            Logger = new ConsoleLogger(),
            Init = init
        };
        var result = finder.TryToFindBestSolution(errorFunction);
        var factory = new ComplexObjectsFactory(result);
        var m1 = factory.CreateMatrix(dimensions, dimensions);
        var m3 = m1 * m2;
        System.Console.WriteLine("Actual inverse matrix");
        System.Console.WriteLine(m2.Inverse());
        System.Console.WriteLine("Converged inverse matrix");
        Console.WriteLine(m1);
        System.Console.WriteLine("Original matrix");
        Console.WriteLine(m2);
        Console.WriteLine("Their product is");
        Console.WriteLine(m3);
        /*
        As you can see, if in actual inverse matrix it's values is way too big, 
        like 14, 25, 10, etc,
        it just only make sense that random matrix that we created will converge way before
        it hits such big values, so it make sense to create a wide range of matrix
        values (like in range from -20, to 20) and try to search local minima in this
        space, yet it is very consuming. 
        So if you at least +- know in which boundaries the actual answer is, I recommend
        to generate initial values for variables in such boundaries
        */
    }
}