using System.Security.Cryptography;
using GradientDescentSharp.NeuralNetwork;
using MathNet.Numerics.Integration;
using MathNet.Numerics.LinearAlgebra.Single;
using ScottPlot;

namespace Playground;
public partial class Examples
{
    public static void SampleComplexDistribution()
    {
        //distribution density
        double f_xy(double x, double y) =>
            0.119918069 * Math.Pow(Math.Sqrt(x) + 2 * y, 2) / (10 + x + y + Math.Exp(x + y));
        double F_xy(double X, double Y) =>
            DoubleExponentialTransformation.Integrate(x =>
            DoubleExponentialTransformation.Integrate(y =>
                    f_xy(x, y),
            0, Y, double.Epsilon),
            0, X, double.Epsilon);

        double loss(IDataAccess<double> guess, double p)
        {
            var l = F_xy(guess[0], guess[1]) - p;
            return l * l;
        }
        double lossGradient(IDataAccess<double> gradient, IDataAccess<double> guess, double p)
        {
            var f = f_xy(guess[0], guess[1]);
            var l = 2 * (F_xy(guess[0], guess[1]) - p) * f;
            return 0;
        }

        var guess = new ArrayDataAccess<double>(2);
        var p = Random.Shared.NextDouble();

        var sampler = new MineDescent(guess, guess => loss(guess, p))
        {
        };
        var defaultFactory = new NNComplexObjectsFactory();
        var layer1 = new Layer(defaultFactory, 2, 64, ActivationFunction.Sigmoid());
        var layer2 = new Layer(defaultFactory, 64, 128, ActivationFunction.Sigmoid());
        var layer3 = new Layer(defaultFactory, 128, 32, ActivationFunction.Sigmoid());
        var layer4 = new Layer(defaultFactory, 32, 1, ActivationFunction.Tanh());
        var nn = new ForwardNN(layer1, layer2, layer3,layer4);

        var sample = ()=>{
            var x = Random.Shared.NextSingle() * 8;
            var y = Random.Shared.NextSingle() * 8;
            var z = (float)f_xy(x, y);
            var input = DenseVector.Create(2, 0);
            input[0] = x;
            input[1] = y;
            var output = DenseVector.Create(1, z);
            return (input,output);
        };
        System.Console.WriteLine("Creating data");
        var train = Enumerable.Range(0, 5000).ToList().AsParallel().Select(_=>sample()).ToArray();
        var test = Enumerable.Range(0, 1000).ToList().AsParallel().Select(_=>sample()).ToArray();
        System.Console.WriteLine("Train....");
        for(int i = 0;i<3;i++){
            foreach(var t in train){
                nn.Backwards(t.input,t.output);
                nn.Backwards(t.input,t.output);
                // nn.LearnOnLoss(t.input,0.01f,(inp,pnn)=>(float)(pnn.Forward(inp)-t.output).Sum(x=>x*x));
            }
        }
        
        var error = test.Select(t=>(nn.Forward(t.input)-t.output)[0]).Sum(x=>x*x)/test.Length;
        System.Console.WriteLine(error);
    }
}