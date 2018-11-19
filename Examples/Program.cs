using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Ann.Activators;
using Ann.Core.LossFunctions;
using ShellProgressBar;

namespace Ann.Mnist
{
    class Program
    {
        static void Main(string[] args)
        {
            var network = CreateModel();
            TrainModel(network, q => Helper.Create3DInput(q.Data));
            var ratio = TestModel(network, q => Helper.Create1DInput(q.Data));
            Console.WriteLine($"Accuracy: {ratio * 100}% ");
            Console.ReadLine();
        }

        private static Network CreateModel()
        {
            var network = new Network(LossFunctionType.CrossEntropy, 10);

            var lr = 0.001;

            network.AddInputLayer(28, 1);
            network.AddConvolutionLayer(Optimizers.Flat(lr), 16, 5);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddPoolingLayer(2);
            network.AddConvolutionLayer(Optimizers.Flat(lr), 32, 5);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddPoolingLayer(2);
            network.AddFlattenLayer();
            network.AddHiddenLayer(512, ActivatorType.Relu, Optimizers.Flat(lr));
            network.AddSoftMaxLayer(Optimizers.Flat(lr));
            network.RandomizeWeights(0.1);

            return network;
        }

        private static void TrainModel(Network model, Func<Image,Array> getInput)
        {
            int total = 60000;
            int current = 0;
            using (var pbar = new ProgressBar(total, "Training Model"))
            {
                foreach (var image in MnistReader.ReadTrainingData(10000))
                {
                    var target = Helper.CreateTarget(image.Label);
                    model.TrainModel(getInput(image), target);
                    pbar.Tick($"Training Model: {++current} of {total}");
                }
            }
        }

        private static void TrainModel(Network model, int batchSize, Func<Image, Array> getInput)
        {
            int total = 10000;
            int current = 0;
            using (var pbar = new ProgressBar(total, "Training Model"))
            {
                List<Task> tasks = new List<Task>();
                foreach (var image in MnistReader.ReadTrainingData(total))
                {
                    if(tasks.Count < batchSize)
                    { 
                        var target = Helper.CreateTarget(image.Label);
                        tasks.Add(Task.Run(() => model.TrainModel(getInput(image), target)));
                    }
                    else
                    {
                        Task.WaitAll(tasks.ToArray());
                        tasks.Clear();
                    }

                    pbar.Tick($"Training Model: {++current} of {total}");
                }
            }
        }

        private static double TestModel(Network model, Func<Image, Array> getInput)
        {
            var results = new List<double>();
            int total = 10000;
            int current = 0;
            using (var pbar = new ProgressBar(total, "Testing Model"))
            {
                foreach (var image in MnistReader.ReadTestData(total))
                {
                    var res = model.UseModel(getInput(image));
                    results.Add(Helper.IntegerFromOutput(res) == image.Label ? 1 : 0);
                    pbar.Tick($"Testing Model: {++current} of {total}");
                }
            }

            return results.Average();
        }
    }
}
