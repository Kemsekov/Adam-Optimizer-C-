using System.Diagnostics;
using System.Globalization;
using CsvHelper;
using GradientDescentSharp.NeuralNetwork;
using Playground.DataModels;

namespace Playground;
public partial class Examples
{
    public static void NeuralNetworkClassificationExample()
    {
        using var reader = new StreamReader("datasets/IRIS.csv");
        using var data = new CsvReader(reader, CultureInfo.InstalledUICulture);

        var records = data.GetRecords<Iris>().ToArray();
        Random.Shared.Shuffle(records);

        var train = records[20..];
        var test = records[..20];

        var layer1 = new Layer(4, 32, ActivationFunction.Tanh());
        var layer2 = new Layer(32, 16, ActivationFunction.LeakyRelu(0.2f));
        var layer3 = new Layer(16, 3, ActivationFunction.Softmax());

        var speciesPredictor = new ForwardNN(layer1, layer2, layer3)
        {
            LearningRate = 0.05f,
            // LearnerFactory=DefaultLearner.Factory(new L2Regularization(0.001f))
        };

        var getError = () => test.Average(x =>
        {
            var data = x.BuildData();
            return speciesPredictor.Error(data.input, data.output);
        });

        var trainData = 
            train
            .Select(t=>t.BuildData())
            .Select(t=>new{
                input=t.input.ToTensor(),
                output=t.output.ToTensor()
            })
            .ToList();
        var timer = new Stopwatch();
        var learnTimer = new Stopwatch();
        timer.Start();
        
        for (int epoch = 0; epoch < 40; epoch++)
        {
            System.Console.WriteLine($"Error is {getError()}");
            trainData.OrderBy(n=>Random.Shared.Next())
            .Chunk(10)
            .ToList()
            .ForEach(chunk=>
                chunk
                .Select(d=>speciesPredictor.Backwards(d.input, d.output))
                .ToList()
                .ForEach(n=>{learnTimer.Start();n.Learn();learnTimer.Stop();})
            );
            
        }
        System.Console.WriteLine($"Final error is {getError()}");
        System.Console.WriteLine("Total time "+timer.ElapsedMilliseconds);
        System.Console.WriteLine("Learn time "+learnTimer.ElapsedMilliseconds);

        for(int i =0;i<5;i++){
            var actual = test[i];
            var prediction = speciesPredictor.Forward(actual.BuildData().input);
            System.Console.WriteLine("------------------");
            System.Console.WriteLine(actual);
            System.Console.WriteLine("Predicted:");
            System.Console.WriteLine($"{Iris.IdToSpeciesMap[0]} :\t\t{prediction[0]}");
            System.Console.WriteLine($"{Iris.IdToSpeciesMap[1]} :\t{prediction[1]}");
            System.Console.WriteLine($"{Iris.IdToSpeciesMap[2]} :\t{prediction[2]}");
        }
    }
}