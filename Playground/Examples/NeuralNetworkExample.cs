namespace Playground;
using GradientDescentSharp.NeuralNetwork;

public partial class Examples
{
    //In this example we learn two continuous function:
    //y1=sin(x1+x2)
    //y2=x1*x2
    public static void NeuralNetworkExample()
    {
        //used to pretty print vectors
        string formatVector(Vector v) => $"{v[0]:0.000}, {v[1]:0.000}";

        var defaultFactory = new NNComplexObjectsFactory();

        //here I decided to show three of weight initializers used as example
        var he = new HeNormal();
        var glorotUniform = new GlorotUniform();
        var glorotNormal = new GlorotNormal();

        var layer1 = new Layer(defaultFactory, 2, 32, ActivationFunction.Tanh(), glorotUniform);
        var layer2 = new Layer(defaultFactory, 32, 16, ActivationFunction.Tanh(), glorotNormal);

        //output layed needs to be linear so both positive and negative values can be 
        //predicted by a model
        var layer3 = new Layer(defaultFactory, 16, 2, ActivationFunction.Linear(), he);

        var nn = new ForwardNN(layer1, layer2, layer3);

        //learning rate is changing dynamically depending on layer weights,
        //so in solution space we step always +- same distance to local minima
        nn.LearningRate = 0.1;
        for (int k = 0; k < 50; k++)
        {
            var error = 0.0;
            for (int i = 0; i < 100; i++)
            {
                var input = DenseVector.Create(2, x => Random.Shared.NextDouble() * 4 - 2);
                var expected = DenseVector.Create(2, 0);
                expected[0] = Math.Sin(input[0] + input[1]);
                expected[1] = input[0] * input[1];

                //this version of backpropagation at it's core support rolling back
                //to original weights, if we hit a worse minima after learning!
                //it can be used to manually decrease learning rate if we hit too much
                //of such failed backpropagations.
                nn.Backwards(input, expected);
                error += nn.Error(input, expected);
            }
            System.Console.WriteLine($"Error is {error / 100}");
        }

        //show some examples of predictions
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