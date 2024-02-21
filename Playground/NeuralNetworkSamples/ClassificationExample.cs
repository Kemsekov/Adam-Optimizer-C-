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

        var defaultFactory = new NNComplexObjectsFactory();
        var layer1 = new Layer(defaultFactory, 4, 32, ActivationFunction.Tanh());
        var layer2 = new Layer(defaultFactory, 32, 16, ActivationFunction.Tanh());
        var layer3 = new Layer(defaultFactory, 16, 3, ActivationFunction.Softmax());

        var speciesPredictor = new ForwardNN(layer1, layer2, layer3)
        {
            LearningRate = 0.05f
        };

        var getError = () => test.Average(x =>
        {
            var data = x.BuildData();
            return speciesPredictor.Error(data.input, data.output);
        });

        for (int epoch = 0; epoch < 40; epoch++)
        {
            System.Console.WriteLine($"Error is {getError()}");
            foreach (var record in train)
            {
                var (input, expected) = record.BuildData();
                var backprop = speciesPredictor.Backwards(input, expected);
                backprop.Learn();
            }
        }
        System.Console.WriteLine($"Final error is {getError()}");

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