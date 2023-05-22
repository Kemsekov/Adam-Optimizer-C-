using System.Globalization;
using CsvHelper;
using GradientDescentSharp.NeuralNetwork;
using Playground.DataModels;
using MathNet.Numerics.LinearAlgebra.Single;

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

        pricePredictor.LearningRate = 0.05f;

        var LossFunction = (UsaHousing record) =>
        {
            var expected = DenseVector.Create(1, (float)record.Price);
            var input = record.ToInputVector();
            return pricePredictor.Error(input, expected);
        };
        var batchSize = 500;
        //here we train
        for (int epoch = 0; epoch < 20; epoch++)
        {
            var testError = test.Average(LossFunction);
            var trainError = 0.0;
            foreach (var record in train.TakeNRandom(batchSize))
            {
                var input = record.ToInputVector();
                var expected = DenseVector.Create(1, (float)record.Price);
                var error = LossFunction(record);
                // var backprop = pricePredictor.Backwards(input, expected);

                //alternatively
                var backprop = pricePredictor.LearnOnLoss(input, 1e-3f, (input, nn) => nn.Error(input, expected));
                backprop.Learn();
                
                var errorAfter = pricePredictor.Error(input, expected);
                trainError += errorAfter;
            }

            trainError /= batchSize;
            System.Console.WriteLine("----------------");
            System.Console.WriteLine($"Train error is {trainError}");
            System.Console.WriteLine($"Test error is {testError}");
        }

        //here we evaluate
        foreach (var sample in test.TakeNRandom(4))
        {
            System.Console.WriteLine("---------------");
            var input = sample.ToInputVector();
            var prediction = pricePredictor.Forward(input)[0];

            var restored = sample.RestoreRecord(meanRecordValues);
            System.Console.WriteLine(restored);
            System.Console.WriteLine($"Predicted price:\t{prediction * meanRecordValues.Price}");
        }
    }
}