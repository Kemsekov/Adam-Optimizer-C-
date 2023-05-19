using System.Globalization;
using CsvHelper;
using GradientDescentSharp.NeuralNetwork;
using Playground.DataModels;

namespace Playground;
public partial class Examples
{
    public static void NeuralNetworkOnDatasetExample()
    {

        var defaultFactory = new NNComplexObjectsFactory();
        var layer1 = new Layer(defaultFactory, 5, 32, ActivationFunction.Tanh(), Initializers.GlorotUniform);
        var layer2 = new Layer(defaultFactory, 32, 16, ActivationFunction.Tanh(), Initializers.GlorotUniform);
        var layer3 = new Layer(defaultFactory, 16, 1, ActivationFunction.Linear(), Initializers.Guassian);

        var pricePredictor = new ForwardNN(layer1, layer2, layer3);
        using var reader = new StreamReader("datasets/USA_Housing.csv");
        using var data = new CsvReader(reader, CultureInfo.InstalledUICulture);
        var records = data.GetRecords<UsaHousing>().ToArray();


        var meanRecordValues = new UsaHousing()
        {
            AreaPopulation = records.Average(x => x.AreaPopulation),
            AvgAreaIncome = records.Average(x => x.AvgAreaIncome),
            AvgHouseAge = records.Average(x => x.AvgHouseAge),
            AvgNumberOfRooms = records.Average(x => x.AvgNumberOfRooms),
            AvgNumberOfBedrooms = records.Average(x => x.AvgNumberOfBedrooms),
            Price = records.Average(x => x.Price)
        };
        foreach (var r in records)
            r.NormalizeRecord(meanRecordValues);

        var train = records[..4500];
        var test = records[4500..];

        pricePredictor.LearningRate = 0.05;

        var errorFunction = (UsaHousing record) =>
        {
            var expected = DenseVector.Create(1, record.Price);
            var input = record.ToInputVector();
            return pricePredictor.Error(input, expected);
        };
        //here we train
        for (int epoch = 0; epoch < 30; epoch++)
        {
            var epochError = test.Sum(errorFunction);
            System.Console.WriteLine($"Error is {epochError}");

            foreach (var record in train.TakeNRandom(500))
            {
                var input = record.ToInputVector();
                var expected = DenseVector.Create(1, record.Price);
                var error = pricePredictor.Error(input, expected);
                var backprop = pricePredictor.Backwards(input, expected);

                //alternatively
                // var backprop = pricePredictor.LearnOnError(input,1e-6,(input,nn)=>nn.Error(input,expected));

                var errorAfter = pricePredictor.Error(input, expected);
                if (errorAfter > error)
                {
                    pricePredictor.LearningRate *= 0.5;
                    backprop.Unlearn();
                    System.Console.WriteLine("Unlearn");
                }
            }

        }

        //here we evaluate
        foreach(var sample in test.TakeNRandom(10)){
            System.Console.WriteLine("---------------");
            var input = sample.ToInputVector();
            var prediction = pricePredictor.Forward(input)[0];

            var restored = sample.RestoreRecord(meanRecordValues);
            System.Console.WriteLine(restored);
            System.Console.WriteLine($"Predicted price:\t{prediction*meanRecordValues.Price}");
        }
    }
}