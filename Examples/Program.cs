using System;
using Ann.Activators;
using Ann.Core;
using Ann.Core.LossFunctions;
using Gdo.Optimizers;
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
            var ratio = Decimal.Divide(res.success,res.success + res.fail);
            Console.WriteLine($"Success: {res.success} | Fail: {res.fail} | {ratio * 100}% ");
            Console.ReadLine();
        }

        private static Network CreateModel()
        {
            var network = new Network(LossFunctionType.CrossEntropy, 10);

            network.AddInputLayer(28, 1);
            network.AddHiddenLayer(16, ActivatorType.Sigmoid, Optimizers.Flat(0.1));
            network.AddHiddenLayer(16, ActivatorType.Sigmoid, Optimizers.Flat(0.1));
            network.AddSoftMaxLayer(Optimizers.Flat(0.1));
            network.RandomizeWeights();

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
                    var data = Helper.CreateInput(image.Data);
                    var target = Helper.CreateTarget(image.Label);
                    model.TrainModel(new Message(data), target);
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
                    var data = Helper.CreateInput(image.Data);
                    var res = model.UseModel(new Message(data));
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
