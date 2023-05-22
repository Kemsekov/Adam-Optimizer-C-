using GradientDescentSharp.NeuralNetwork;
using ScottPlot;
using MathNet.Numerics.LinearAlgebra.Single;
using GradientDescentSharp.NeuralNetwork.Specific;

namespace Playground;
public partial class Examples
{
    public static void FindComplexFunction()
    {
        // we need to find such function f(x), that
        // f(x)+f(e^x+0.1)+1=0

        //this problem is unsolvable by ordinary backpropagation Backwards method,
        //because we don't have any dataset, but it is solvable by error function!

        float Problem(Vector input, PredictOnlyNN nn)
        {
            var x = input[0];
            var y1 = nn.Forward(input)[0];
            var y2 = nn.Forward((Vector)input.Map(x => MathF.Exp(x)+0.1f))[0];
            var error = y1 + y2+1;
            return error * error;
        };

        var defaultFactory = new NNComplexObjectsFactory();
        var layer1 = new Layer(defaultFactory, 1, 24, ActivationFunction.Tanh(), Initializers.GlorotUniform);
        var layer2 = new Layer(defaultFactory, 24, 16, ActivationFunction.Tanh(), Initializers.GlorotUniform);
        var layer3 = new Layer(defaultFactory, 16, 1, ActivationFunction.Linear(), Initializers.Guassian);
        var nn = new ForwardNN(layer1, layer2, layer3);

        var testInputs = Enumerable.Range(-10, 10).Select(x => DenseVector.Create(1, x * 1.0f / 20)).ToArray();

        nn.LearningRate = 0.05f;
        for (int epoch = 0; epoch < 10; epoch++)
        {

            for (int i = 0; i < 500; i++)
            {
                var x = Random.Shared.NextSingle() * 4 - 2;
                var input = DenseVector.Create(1, x);
                var result = nn.LearnOnLoss(input, 0.01f, Problem);
                result.Learn();
            }
            var epochError = testInputs.Sum(x => Problem(x, nn));
            System.Console.WriteLine("Error " + epochError);
        }

        var plt = new ScottPlot.Plot();
        var dots = 100;
        var xValues = new double[dots];
        var yValues = new double[dots];
        for (int i = 0; i < dots; i++)
        {
            var x = 4 * (i - dots / 2) * 1.0f / dots;
            var input = DenseVector.Create(1, x);
            var y1 = nn.Forward(input)[0];
            var y2 = nn.Forward((Vector)input.Map(x => MathF.Exp(-x)))[0];
            xValues[i] = x;
            yValues[i] = y1;
        }
        plt.Add.Scatter(xValues, yValues, System.Drawing.Color.Red.ToScatter());
        plt.SaveJpeg("s.jpg", 1000, 1000);
        for(int i = 0;i<5;i++){
            var x1 = Random.Shared.NextSingle() * 4 - 2;
            var x2 = MathF.Exp(x1)+0.1f;
            var y1 = nn.Forward(DenseVector.Create(1,x1))[0];
            var y2 = nn.Forward(DenseVector.Create(1,x2))[0];
            System.Console.WriteLine("---------------------");
            System.Console.WriteLine($"x:\t\t\t{x1:0.00}");
            System.Console.WriteLine($"e^x+0.1:\t\t{x2:0.00}");
            System.Console.WriteLine($"f(x)+f(e^x+0.1)+1 =\t{y1+y2+1:0.00}");
        }
    }
}