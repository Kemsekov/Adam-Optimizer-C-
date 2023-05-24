namespace Playground;
using MathNet.Numerics.LinearAlgebra.Single;

using GradientDescentSharp.NeuralNetwork;

public partial class Examples
{
    //In this example we learn two continuous function:
    //y1=sin(x1+x2)
    //y2=x1*x2
    public static void NeuralNetworkContinuousLearningExample()
    {
        var defaultFactory = new NNComplexObjectsFactory();

        var layer1 = new Layer(defaultFactory, 2, 32, ActivationFunction.Tanh(), Initializers.GlorotUniform);
        var layer2 = new Layer(defaultFactory, 32, 16, ActivationFunction.Tanh(), Initializers.GlorotNormal);

        //output layed needs to be linear so both positive and negative values can be 
        //predicted by a model
        var layer3 = new Layer(defaultFactory, 16, 2, ActivationFunction.Linear(), Initializers.Guassian);

        var nn = new ForwardNN(layer1, layer2, layer3);

        for (int i = 0; i < 5; i++)
        {
            var y1 = (float x1, float x2) => MathF.Sin(x1 + x2);
            var y2 = (float x1, float x2) => x1 * x2;
            _NeuralNetworkExample(nn, y1, y2);
            System.Console.WriteLine("------------------Change functions!------------------");
            y1 = (float x1, float x2) => MathF.Cos(x1 - x2);
            y2 = (float x1, float x2) => x1 - x2 * MathF.Sin(x1);
            _NeuralNetworkExample(nn, y1, y2);
            System.Console.WriteLine("------------------Epoch end------------------");
        }

    }
    public static void NeuralNetworkExample()
    {
        var defaultFactory = new NNComplexObjectsFactory();

        var layer1 = new Layer(defaultFactory, 2, 32, ActivationFunction.Tanh(), Initializers.GlorotUniform);
        var layer2 = new Layer(defaultFactory, 32, 16, ActivationFunction.Tanh(), Initializers.GlorotNormal);

        //output layed needs to be linear so both positive and negative values can be 
        //predicted by a model
        var layer3 = new Layer(defaultFactory, 16, 2, ActivationFunction.Linear(), Initializers.Guassian);

        var nn = new ForwardNN(layer1, layer2, layer3);

        var y1 = (float x1, float x2) => MathF.Sin(x1 + x2);
        var y2 = (float x1, float x2) => x1 * x2;
        _NeuralNetworkExample(nn, y1, y2);


    }
    static void _NeuralNetworkExample(NNBase nn, Func<float, float, float> y1, Func<float, float, float> y2)
    {
        //used to pretty print vectors
        string formatVector(Vector v) => $"{v[0]:0.000}, {v[1]:0.000}";

        //learning rate is changing dynamically depending on layer weights,
        //so in solution space we step always +- same distance to local minima

        //here I purposely start with a higher learning rate, than it should be
        var testData = Enumerable.Range(0, 100).Select(x => DenseVector.Create(2, x => Random.Shared.NextSingle() * 4 - 2)).ToArray();
        nn.LearningRate = 0.01f;
        for (int k = 0; k <= 40; k++)
        {
            LearnFunctionsBody(nn, y1, y2, testData, k);
        }

        //show some examples of predictions
        for (int i = 0; i < 5; i++)
        {
            var num1 = Random.Shared.NextSingle() * 4 - 2;
            var num2 = Random.Shared.NextSingle() * 4 - 2;

            var expected = DenseVector.Create(2, 0);
            expected[0] = y1(num1, num2);
            expected[1] = y2(num1, num2);
            var input = DenseVector.Create(2, 0);
            input[0] = num1;
            input[1] = num2;
            
            var result = nn.Forward(input);

            System.Console.WriteLine("-------------");
            System.Console.WriteLine($"Expected: {formatVector(expected)}");
            System.Console.WriteLine($"Result: {formatVector(result)}");
        }

        nn.RegenerateSaturatedWeights(10, 0.01f, 0.99f, 0.2f);
        var replaced = testData.Average(x =>
        {
            var expected = DenseVector.Create(2, 0);
            expected[0] = y1(x[0], x[1]);
            expected[1] = y2(x[0], x[1]);
            return nn.Error(x, expected);
        });
        System.Console.WriteLine("Test error after replacing saturated " + replaced);
    }

    private static void LearnFunctionsBody(NNBase nn, Func<float, float, float> y1, Func<float, float, float> y2, DenseVector[] testData, int k)
    {
        var error = 0.0;
        for (int i = 0; i < 100; i++)
        {
            var input = DenseVector.Create(2, x => Random.Shared.NextSingle() * 4 - 2);
            var expected = DenseVector.Create(2, 0);
            expected[0] = y1(input[0], input[1]);
            expected[1] = y2(input[0], input[1]);


            //this version of backpropagation at it's core support rolling back
            //to original weights, if we hit a worse minima after learning!
            //it can be used to manually decrease learning rate if we hit too much
            //of such failed backpropagations.

            //we also can learn on error function instead. Uncomment it to see
            var backprop = nn.Backwards(input, expected);
            // var backprop = nn.LearnOnLoss(input, 1e-3f, (input1, nn1) => (nn1.Forward(input1) - expected).Sum(x => x * x));

            var beforeLearn = nn.Error(input, expected);
            backprop.Learn();
            var afterLearn = nn.Error(input, expected);

            //when we hit a worsen rather than improvement, 
            //it indicates that our learning rate is too high, so we 
            //undo changes from previous learning and decrease learning rate
            if (afterLearn > beforeLearn || float.IsNaN(afterLearn))
            {
                backprop.Unlearn();
                nn.LearningRate *= 0.9f;
                var unlearnedError = nn.Error(input, expected);
                System.Console.WriteLine("Unlearn");
            }
            error += afterLearn;
        }
        if (k % 10 == 0)
        {
            var testError = testData.Average(x =>
            {
                var expected = DenseVector.Create(2, 0);
                expected[0] = y1(x[0], x[1]);
                expected[1] = y2(x[0], x[1]);
                return nn.Error(x, expected);
            });
            System.Console.WriteLine("-----------------------");
            System.Console.WriteLine($"Test error is {testError}");
            System.Console.WriteLine($"Train error is {error / 100}");
        }
    }
}
