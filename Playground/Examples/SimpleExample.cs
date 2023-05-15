namespace Playground;
public partial class Examples
{
    public static void SimpleExample()
    {
        //first define a problem
        var problem = (IDataAccess<double> x) =>
        {
            var n = x[0];
            //we seek for such value, that n^n=5
            var needToMinimize = Math.Pow(n, n) - 5.0;
            return Math.Abs(needToMinimize);
        };
        //then define changing variables
        var variables = new ArrayDataAccess<double>(1);
        //set variables close to global minima
        variables[0] = 1;

        //define descent
        var descent = new MineDescent(variables, problem)
        {
            DescentRate = 0.1,             // how fast to descent, this value will be adjusted on the fly
            Theta = 1e-4,                  // what precision of output we need
            DescentRateDecreaseRate = 0.1, // how much decrease DescentRate when we hit a grow of error function
            Logger = new ConsoleLogger()   // logger for descent progress
        };

        //do 30 iterations
        descent.Descent(30);

        System.Console.WriteLine("For problem n^n=5");
        System.Console.WriteLine($"Error is {problem(variables)}");
        System.Console.WriteLine($"n={variables[0]}");
        System.Console.WriteLine($"n^n={Math.Pow(variables[0], variables[0])}");
    }
}