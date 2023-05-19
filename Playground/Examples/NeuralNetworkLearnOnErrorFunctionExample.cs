namespace Playground;
using GradientDescentSharp.NeuralNetwork;

public partial class Examples
{
    //in this example we can see, how neural network can actually learn
    //from error function only. That's it!
    public static void NeuralNetworkLearnOnErrorFunctionExample()
    {
        var defaultFactory = new NNComplexObjectsFactory();

        var layer1 = new Layer(defaultFactory, 1, 32, ActivationFunction.Tanh(), Initializers.GlorotUniform);
        var layer2 = new Layer(defaultFactory, 32, 16, ActivationFunction.Tanh(), Initializers.GlorotUniform);
        var layer3 = new Layer(defaultFactory, 16, 4, ActivationFunction.Tanh(), Initializers.GlorotUniform);
        var layer4 = new Layer(defaultFactory, 4, 1, ActivationFunction.Linear(), Initializers.Guassian);

        var nn = new ForwardNN(layer1, layer2, layer3, layer4);
        _NeuralNetworkLearnOnErrorFunctionExample(nn);
    }
    public static void _NeuralNetworkLearnOnErrorFunctionExample(NNBase nn){
        var xValues = Enumerable.Range(0, 1000).Select(x => DenseVector.Create(1, Random.Shared.NextDouble() * 4)).ToArray();
        //Here we define a problem, which is error function.
        var problem = (Vector input, NNBase nn) =>
        {
            var result = nn.Forward(input);
            //we really need to approximate x^2.5 function
            return Math.Pow(result[0] - Math.Pow(input[0], 2.5), 2);
        };

        nn.LearningRate = 0.01;
        var error = 0.0;
        for(var epoch = 0;epoch<5;epoch++)
        for (int i = 0; i < xValues.Length; i++)
        {
            var x = xValues[i];
            var expected = DenseVector.Create(1, Math.Pow(x[0], 2.5));
            var errorBefore = nn.Error(x, expected);

            // here magic happens. The some information we put into is:
            // input vector, some small theta, that will be used to compute
            // derivatives, and error function. This is enough to learn!
            var backprop = nn.LearnOnError(x, 0.0001, problem);

            var errorAfter = nn.Error(x, expected);

            // after we learnt something, we need to check if we didn't messed up
            // if new error is bigger, than previous one, we rollback changes to
            // neural network!
            if (errorAfter > errorBefore)
            {
                nn.LearningRate *= 0.5;
                backprop.Unlearn();
                System.Console.WriteLine("Unlearn");
            }
            error += errorAfter;
            if (i % 100 == 0 && i != 0)
            {
                System.Console.WriteLine("Error : " + error / 100);
                error = 0.0;
            }
        }
        var testError = 0.0;
        //print some values for prediction
        for (int i = 0; i < 10; i++)
        {
            var x1 = DenseVector.Create(1, i / 3.0);
            var x = x1[0];
            var xPower2dot5 = Math.Pow(x1[0], 2.5);
            var predicted = nn.Forward(x1)[0];
            System.Console.WriteLine("------------------");
            System.Console.WriteLine($"x={x:0.0000}");
            System.Console.WriteLine($"x^2.5={xPower2dot5:0.0000}");
            System.Console.WriteLine($"Predicted={predicted:0.0000}");
            testError += Math.Abs(predicted-xPower2dot5);
        }
        System.Console.WriteLine("Test avg error is "+testError/10);
    }
}