namespace Playground;
using MathNet.Numerics.LinearAlgebra.Single;

using GradientDescentSharp.NeuralNetwork;
using MathNet.Numerics.LinearAlgebra;
using ScottPlot;

public partial class Examples
{
    public static void Regularization()
    {
        float[][] X = [
            [1,2],
            [3,4],
            [4,4],
            [6,2],
            [6,1],
            [8,3],
            [7,3]
        ];

        float[][] Y = [
            [255,255],
            [235,180],
            [200,150],
            [170,100],
            [220,180],
            [120,90],
            [0,0]
        ];

        var plt = new Plot();
        var data = X.Select(DenseVector.OfArray).Zip(Y.Select(DenseVector.OfArray)).ToArray();


        var layer1 = new Layer(2, 16, ActivationFunction.Tanh());
        var layer2 = new Layer(16, 4, ActivationFunction.Tanh());
        var layer3 = new Layer(4, 2, ActivationFunction.LeakyRelu(0.3f));

        var nn = new ForwardNN(layer1, layer2, layer3)
        {
            LearningRate = 0.01f,
            LearnerFactory = DefaultLearner.Factory(new L2Regularization(0.01f))
        };

        for (int epoch = 0; epoch < 1000; epoch++)
        {
            //we first compute backward operations and then apply learn on all of them at once
            data
            .Select(row=>nn.Backwards(row.First,row.Second/100))
            .ToList()
            .ForEach(n=>n.Learn());
        }

        //draw predictions
        for (int i = 0; i < 100; i++)
        {
            for (int j = 0; j < 100; j++)
            {
                float x = i * 1.0f / 100 * 7 + 1;
                float y = j * 1.0f / 100 * 5;
                var color = nn.Forward(DenseVector.OfArray([x, y])) * 100;
                color.MapInplace(t => Math.Min(t, 255));
                color.MapInplace(t => Math.Max(t, 0));
                if (color.Any(float.IsNaN)) continue;
                plt.Add.Scatter(
                x, y,
                System.Drawing.Color.FromArgb((int)color[0], (int)color[1], 0).ToScatter()).MarkerSize = 10;
            }
        }

        //draw dots
        foreach (var (x, y) in data)
        {
            plt.Add.Scatter(
                x[0], x[1],
                System.Drawing.Color.FromArgb((int)y[0], (int)y[1], 0).ToScatter()).MarkerSize = 15;
        }

        plt.SaveJpeg("res.jpg", 1000, 1000);
    }
}
