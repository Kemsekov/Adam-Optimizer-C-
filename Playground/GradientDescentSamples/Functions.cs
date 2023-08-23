namespace Playground;
public partial class Examples
{
    /// <summary>
    /// Finds parameters for cannon ball. You can set them and see how it works on calculator
    /// https://www.desmos.com/calculator/qe8pvhg7ur
    /// </summary>
    public static void CannonBallPathParameters()
    {
        double[] dataX = new double[] { 0.01, 0.06, 0.11, 0.21, 0.31, 0.46, 0.61, 0.71, 0.80, 0.85, 0.02, 0.07, 0.12, 0.16, 0.22, 0.28, 0.35, 0.45, 0.55, 0.66, 0.76, 0.82, 0.84, 0.04, 0.08, 0.14, 0.18, 0.25, 0.33, 0.42, 0.50, 0.56, 0.61, 0.63, 0.66, 0.04, 0.10, 0.15, 0.21, 0.28, 0.35, 0.43, 0.50, 0.57, 0.63, 0.68, 0.72, 0.76, 0.02, 0.07, 0.12, 0.17, 0.22, 0.28, 0.36, 0.42, 0.53, 0.62, 0.69, 0.74, 0.78, 0.72, 0.67, 0.62, 0.80, 0.75, 0.69, 0.64, 0.56, 0.46, 0.36, 0.28, 0.18, 0.12, 0.07, 0.04, 0.02, 0.59, 0.53, 0.44, 0.37, 0.26, 0.18, 0.13, 0.08, 0.04, -0.00, 0.01, 0.91, 0.86, 0.82, 0.84, 0.77, 0.78, 0.72, 0.72, 0.70, 0.65, 0.61, 0.55, 0.48, 0.42, 0.30, 0.24, 0.15, 0.09, 0.04, 0.69, 0.66, 0.62, 0.58, 0.54, 0.81, 0.75, 0.71, 0.62, 0.54, 0.48, 0.40, 0.32, 0.22, 0.16, 0.12, 0.07 };
        double[] dataY = new double[] { 0.03, 0.09, 0.17, 0.23, 0.27, 0.28, 0.24, 0.19, 0.11, 0.05, 0.06, 0.13, 0.20, 0.25, 0.28, 0.30, 0.32, 0.32, 0.31, 0.28, 0.23, 0.13, 0.05, 0.03, 0.10, 0.17, 0.22, 0.24, 0.26, 0.25, 0.21, 0.18, 0.12, 0.06, 0.02, 0.06, 0.14, 0.20, 0.24, 0.26, 0.27, 0.28, 0.27, 0.22, 0.19, 0.14, 0.08, 0.01, 0.03, 0.09, 0.15, 0.20, 0.25, 0.26, 0.27, 0.28, 0.25, 0.22, 0.17, 0.10, 0.04, 0.01, 0.08, 0.15, 0.05, 0.16, 0.22, 0.25, 0.29, 0.30, 0.30, 0.29, 0.26, 0.18, 0.12, 0.08, 0.03, 0.16, 0.22, 0.26, 0.27, 0.26, 0.22, 0.16, 0.10, 0.06, 0.00, 0.03, 0.02, 0.01, 0.00, 0.02, 0.09, 0.09, 0.04, 0.10, 0.15, 0.19, 0.22, 0.25, 0.27, 0.28, 0.27, 0.26, 0.20, 0.13, 0.05, 0.01, 0.08, 0.15, 0.20, 0.24, 0.03, 0.11, 0.16, 0.21, 0.25, 0.27, 0.27, 0.27, 0.25, 0.22, 0.16, 0.11 };

        var variables = new ArrayDataAccess<double>(new double[4]);

        variables[0] = 1;
        variables[1] = -5;
        variables[2] = 3;
        variables[3] = 8;

        double X(double m1, double w, double t) => (m1 * Math.Exp(t) + w) * t;
        double Y(double m2, double g, double t) => (m2 * Math.Exp(t) + g) * t;
        double X_dt(double m1, double w, double t) => m1 * Math.Exp(t) * (1 + t) + w;
        double Y_dt(double m2, double g, double t) => m2 * Math.Exp(t) * (1 + t) + g;
        double loss_t(double m1, double m2, double w, double g, double t, double x0, double y0)
        {
            var x_loss = x0 - X(m1, w, t);
            var y_loss = y0 - Y(m2, g, t);
            return x_loss * x_loss + y_loss * y_loss;
        }
        double loss_dt(double m1, double m2, double w, double g, double t, double x0, double y0)
        {
            var x_loss = x0 - X(m1, w, t);
            var y_loss = y0 - Y(m2, g, t);
            return -2 * (x_loss * X_dt(m1, w, t) + y_loss * Y_dt(m2, g, t));
        }

        double find_t(double m1, double m2, double w, double g, double x0, double y0)
        {
            var t = new ArrayDataAccess<double>(new double[1]);
            t[0] = 1;
            var descent = new MineDescent(t, x => loss_t(m1, m2, w, g, x[0], x0, y0))
            {
                Theta = 0,
                ComputeGradient = (gradient, x, _) => gradient[0] = loss_dt(m1, m2, w, g, x[0], x0, y0)
            };
            descent.Descent(40);
            return t[0];
        }

        double find_v0(double m1, double m2, double w, double g)
        {
            return Math.Sqrt(Math.Pow(m1 + w, 2) + Math.Pow(m2 + g, 2));
        }

        double find_alpha(double m1, double m2, double w, double g)
        {
            return Math.Acos((m1 + w) / find_v0(m1, m2, w, g));
        }

        // new MineDescent(variables,)

        double loss(IDataAccess<double> vars)
        {
            double totalLoss = 0;
            foreach (var (x, y) in dataX.Zip(dataY))
            {
                var t = find_t(vars[0], vars[1], vars[2], vars[3], x, y);
                totalLoss += loss_t(vars[0], vars[1], vars[2], vars[3], t, x, y);
            }
            return totalLoss;
        }

        var descent = new MineDescent(variables, loss)
        {
            Logger = new ConsoleLogger(),
            DescentRate = 1,
            Theta = 0
        };

        descent.Descent(100);

        Console.WriteLine("v0\t" + find_v0(variables[0], variables[1], variables[2], variables[3]));
        Console.WriteLine("alpha\t" + find_alpha(variables[0], variables[1], variables[2], variables[3]));
        Console.WriteLine("w\t" + variables[2]);
        Console.WriteLine("g\t" + variables[3]);
    }
    /// <summary>
    /// Just a set of functions that can be minimized
    /// </summary>
    public static Func<IDataAccess<double>, double>[] Functions ={
                (d)=>Math.Abs(d[0]-d[1]*d[2]+Math.Exp(d[0]*d[1]*d[2])-Math.Sin(d[0]+d[1]+d[2])),
                (d)=>Math.Abs(d[0]-d[1]*d[2]),
                (d)=>Math.Abs(d[0]*d[1]*d[2]-d[0]*Math.Sin(d[2])+Math.Cos(d[0]*d[2])+d[2]),
                (d)=>Math.Abs(d[0]*Math.Cosh(d[0]*d[1]+Math.Sin(d[2]))),
                (d)=>Math.Abs(Math.Tan(d[0]*d[1]+d[2])-Math.Sin(d[1]*d[2]+d[0])),
                (d)=>Math.Abs(Math.Pow(d[0]*d[0],d[1])-d[2]*Math.Exp(d[0])*Math.Log(d[2]*d[2])),
            };
}