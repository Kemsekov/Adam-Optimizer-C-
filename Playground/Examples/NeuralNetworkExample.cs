namespace Playground;
using GradientDescentSharp.NeuralNetwork;

public partial class Examples
{
    public static void NeuralNetworkExample()
    {
        var defaultFactory = new NNComplexObjectsFactory();
        var he = new HeNormal();
        var glorotUniform = new GlorotUniform();
        var glorotNormal = new GlorotNormal();
        var layer1 = new Layer(defaultFactory, 2, 32, ActivationFunction.Tanh(), glorotUniform);
        var layer2 = new Layer(defaultFactory, 32, 16, ActivationFunction.Tanh(), glorotUniform);
        var layer3 = new Layer(defaultFactory, 16, 2, ActivationFunction.Linear(), he);

        var nn = new ForwardNN(layer1, layer2, layer3);
        nn.LearningRate = 0.05;
        var inputs = Enumerable.Range(0, 100).Select(x => DenseVector.Create(2, t => Random.Shared.NextDouble() * 4 - 2)).ToArray();
        for (int k = 0; k < 50; k++)
        {
            var error = 0.0;
            for (int i = 0; i < 100; i++)
            {
                // var input = DenseVector.Create(2,x=>Random.Shared.NextDouble()*4-2);
                var input = inputs[i];
                var expected = DenseVector.Create(2, 0);
                expected[0] = Math.Sin(input[0] + input[1]);
                expected[1] = input[0] * input[1];
                var backprop = nn.Backwards(input, expected);
                error += nn.Error(input,expected);
            }
            System.Console.WriteLine($"Error is {error / 100}");
        }
        string formatVector(Vector v) => $"{v[0]:0.000}, {v[1]:0.000}";
        for (int i = 0; i < 5; i++)
        {
            var num1 = Random.Shared.NextDouble() * 4 - 2;
            var num2 = Random.Shared.NextDouble() * 4 - 2;

            var expected = DenseVector.Create(2, 0);
            expected[0] = Math.Sin(num1 + num2);
            expected[1] = num1 * num2;
            var input = DenseVector.Create(2, 0);
            input[0] = num1;
            input[1] = num2;
            var result = nn.Forward(input);

            System.Console.WriteLine("-------------");
            System.Console.WriteLine($"Expected: {formatVector(expected)}");
            System.Console.WriteLine($"Result: {formatVector(result)}");
        }
    }
}