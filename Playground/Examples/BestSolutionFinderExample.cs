namespace Playground;
public partial class Examples
{
    public static void BestSolutionFinderExample()
    {
        //example of a problem that does not have an exact solution
        var problem = (IDataAccess<double> variables) =>
        {
            var x = variables[0];
            var y = variables[1];
            var z = variables[2];

            var firstError = x * x + y * y + z * z - 1;
            var secondError = x * x * x + y * y * y + z * z * z - 2;
            var thirdError = x * x * x * x + y * y * y * y + z * z * z * z - 3;

            firstError = Math.Pow(firstError, 2);
            secondError = Math.Pow(secondError, 2);
            thirdError = Math.Pow(thirdError, 2);

            return firstError + secondError + thirdError;
        };

        var variables = new ArrayDataAccess<double>(3);

        //best solutions finder will generate a lot of solutions and take the best one
        //so we need to write a init function for their weights, so
        //descent can find a good local minima for error function
        var init = (IDataAccess<double> variables) =>
        {
            for (int i = 0; i < variables.Length; i++)
            {
                variables[i] = Random.Shared.NextDouble() * 4 - 2;
            }
        };
        var finder = new BestSolutionFinder(
            variablesLength: variables.Length,
            x => new MineDescent(x, problem),
            init: init,
            descentIterations: 30,
            solutionsCount: 500
        )
        {
            Logger = new ConsoleLogger()
        };
        var solution = finder.TryToFindBestSolution(problem);
        System.Console.WriteLine("Solution is");
        System.Console.WriteLine($"x : {solution[0]}");
        System.Console.WriteLine($"y : {solution[1]}");
        System.Console.WriteLine($"z : {solution[2]}");

        System.Console.WriteLine($"Error is {problem(solution)}");
    }
}