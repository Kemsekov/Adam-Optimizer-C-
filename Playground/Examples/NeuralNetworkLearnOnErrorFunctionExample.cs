namespace Playground;
using GradientDescentSharp.NeuralNetwork;

public partial class Examples
{
    public static void NeuralNetworkLearnOnErrorFunctionExample()
    {
        var defaultFactory = new NNComplexObjectsFactory();
        var he = new HeNormal();
        var glorotUniform = new GlorotUniform();
        var glorotNormal = new GlorotNormal();

        var layer1 = new Layer(defaultFactory, 1, 32, ActivationFunction.Tanh(), glorotUniform);
        var layer2 = new Layer(defaultFactory, 32, 16, ActivationFunction.Tanh(), glorotUniform);
        var layer3 = new Layer(defaultFactory, 16, 4, ActivationFunction.Tanh(), glorotUniform);
        var layer4 = new Layer(defaultFactory, 4, 1, ActivationFunction.Linear(), he);

        var nn = new ForwardNN(layer1, layer2, layer3, layer4);
        var xValues = Enumerable.Range(0, 3000).Select(x => DenseVector.Create(1, Random.Shared.NextDouble() * 4)).ToArray();


        var problem = (Vector input, NNBase nn) =>
        {
            var result = nn.Forward(input);
            return Math.Pow(result[0] - Math.Pow(input[0], 2.5), 2);
        };

        nn.LearningRate = 0.01;
        var error = 0.0;
        for (int i = 0; i < xValues.Length; i++)
        {
            var x = xValues[i];
            var expected = DenseVector.Create(1, Math.Pow(x[0], 2.5));
            var errorBefore = nn.Error(x, expected);
            var backprop = nn.LearnOnError(x, 0.00001, problem);
            var errorAfter = nn.Error(x, expected);
            if (errorAfter > errorBefore)
            {
                nn.LearningRate *= 0.5;
                backprop.Unlearn();
                var afterUnlearn = nn.Error(x, expected);
                System.Console.WriteLine("We unlearn");
            }
            error += errorAfter;
            if (i % 100 == 0 && i != 0)
            {
                System.Console.WriteLine("Error : " + error / 100);
                error = 0.0;
            }
        }
        for (int i = 0; i < 8; i++)
        {
            var x1 = DenseVector.Create(1, i / 2.0);
            System.Console.WriteLine("------------------");
            System.Console.WriteLine($"x={x1[0]}");
            System.Console.WriteLine($"x^2.5={Math.Pow(x1[0], 2.5)}");
            System.Console.WriteLine($"Predicted={nn.Forward(x1)[0]}");
        }

    }
}