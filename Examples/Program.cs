using System;
using System.Collections.Generic;
using ShellProgressBar;
using Ann.Activators;
using Ann.CostFunctions;

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
            var network = new Network(CostFunctionType.Quadratic, 784);
            network.AddFullyConnectedLayer(
                16, ActivatorType.Sigmoid, LearningRateAnnealerType.Adagrad);
            network.AddFullyConnectedLayer(
                16, ActivatorType.Sigmoid, LearningRateAnnealerType.Adagrad);
            network.AddFullyConnectedLayer(
                10, ActivatorType.Sigmoid, LearningRateAnnealerType.Adagrad);

            return network;

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
                    model.TrainModel(data.ToArray(), target.ToArray());
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
                    var res = model.UseModel(data.ToArray());
                    int predicted = Helper.IntegerFromOutput(new List<double>(res));

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
