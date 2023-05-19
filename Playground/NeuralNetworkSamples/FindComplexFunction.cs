using GradientDescentSharp.NeuralNetwork;
using ScottPlot;

namespace Playground;
public partial class Examples
{
    public static void FindComplexFunction()
    {
        // we need to find such function f(x), that
        // f(x)+f(e^(-x))=0

        //this problem is unsolvable by ordinary backpropagation Backwards method,
        //because we don't have any dataset, but it is solvable by error function!

        double Problem(Vector input, NNBase nn)
        {
            var x = input[0];
            var y1 = nn.Forward(input);
            var y2 = nn.Forward((Vector)input.Map(x => Math.Exp(-x)));
            var error = y1 + y2;
            return error * error;
        };

        var defaultFactory = new NNComplexObjectsFactory();
        var layer1 = new Layer(defaultFactory, 1, 24, ActivationFunction.Tanh(), Initializers.GlorotUniform);
        var layer2 = new Layer(defaultFactory, 24, 16, ActivationFunction.Tanh(), Initializers.GlorotUniform);
        var layer3 = new Layer(defaultFactory, 16, 1, ActivationFunction.Linear(), Initializers.Guassian);
        var nn = new ForwardNN(layer1, layer2, layer3);

        var testInputs = Enumerable.Range(-10, 10).Select(x => DenseVector.Create(1, x * 1.0 / 20)).ToArray();

        nn.LearningRate = 0.05;
        for (int epoch = 0; epoch < 20; epoch++)
        {
            var epochError = testInputs.Sum(x => Problem(x, nn));
            System.Console.WriteLine("Error " + epochError);
            for (int i = 0; i < 500; i++)
            {
                var x = Random.Shared.NextDouble() * 4 - 2;
                var input = DenseVector.Create(1, x);
                var result = nn.LearnOnError(input, 0.0001, Problem);
            }
        }

        var plt = new ScottPlot.Plot();
        var dots = 100;
        var xValues = new double[dots];
        var yValues = new double[dots];
        for (int i = 0; i < dots; i++)
        {
            var x = 4 * (i - dots / 2) * 1.0 / dots;
            var input = DenseVector.Create(1, x);
            var y1 = nn.Forward(input)[0];
            var y2 = nn.Forward((Vector)input.Map(x => Math.Exp(-x)))[0];
            xValues[i] = x;
            yValues[i] = y1;
        }
        plt.Add.Scatter(xValues, yValues, System.Drawing.Color.Red.ToScatter());
        plt.SaveJpeg("s.jpg", 1000, 1000);

    }
}