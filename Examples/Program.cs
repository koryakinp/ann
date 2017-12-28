using Ann.Configuration;
using System;
using System.Collections.Generic;
using ShellProgressBar;

namespace Ann.Mnist
{
    class Program
    {
        static void Main(string[] args)
        {
            var network = CreateModel();
            TrainModel(network);
            var res = TestModel(network);
            Console.WriteLine($"Success: {res.success} | Fail: {res.fail}");
            Console.ReadLine();
        }

        private static Network CreateModel()
        {
            var layerConfig = new LayerConfiguration();

            layerConfig
                .AddInputLayer(784)
                .AddHiddenLayer(16, ActivatorType.LogisticActivator)
                .AddHiddenLayer(16, ActivatorType.LogisticActivator)
                .AddOutputLayer(10, ActivatorType.LogisticActivator);

            var config = new NetworkConfiguration(layerConfig)
            {
                Momentum = 0.9,
                LearningRate = 0.1
            };

            return new Network(config);
        }

        private static void TrainModel(Network model)
        {
            int total = 60000;
            int current = 0;
            using (var pbar = new ProgressBar(total, "Training Model"))
            {
                foreach (var image in MnistReader.ReadTrainingData())
                {
                    List<double> data = Helper.CreateInput(image.Data);
                    List<double> target = Helper.CreateTarget(image.Label);
                    model.TrainModel(data, target);
                    pbar.Tick($"Training Model: {++current} of {total}");
                }
            }
        }

        private static (int success, int fail) TestModel(Network model)
        {
            int success = 0;
            int fail = 0;

            int total = 10000;
            int current = 0;
            using (var pbar = new ProgressBar(total, "Testing Model"))
            {
                foreach (var image in MnistReader.ReadTestData())
                {
                    List<double> data = Helper.CreateInput(image.Data);
                    var res = model.UseModel(data);
                    int predicted = Helper.IntegerFromOutput(res);

                    if (predicted == image.Label)
                    {
                        success++;
                    }
                    else
                    {
                        fail++;
                    }

                    pbar.Tick($"Testing Model: {++current} of {total}");
                }
            }

            return (success, fail);
        }
    }
}
