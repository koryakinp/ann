using Ann.Activators;
using Ann.LossFunctions;
using Gdo.Optimizers;
using ShellProgressBar;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Ann.Fingers
{
    class Program
    {
        private static readonly int NumberOfClasses = 6;

        static void Main(string[] args)
        {
            var network = CreateModel();
            TrainModel(network, q => Helper.Create3DInput(q.Data));
            //var model = new Model("model.json");
            var model = network.BuildModel();
            network.BuildModel();

            var ratio = TestModel(model, q => Helper.Create3DInput(q.Data));
            Console.WriteLine($"Accuracy: {ratio * 100}% ");
            Console.ReadLine();
            //model.Save("model.json");
        }

        private static Network CreateModel()
        {
            var network = new Network(LossFunctionType.CrossEntropy, new Flat(0.001), NumberOfClasses);

            network.AddInputLayer(128, 1);
            network.AddConvolutionLayer(8, 5);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddPoolingLayer(2);
            network.AddConvolutionLayer(8, 8);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddPoolingLayer(2);
            network.AddFlattenLayer();
            network.AddDenseLayer(64, true);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddDenseLayer(NumberOfClasses, false);
            network.AddSoftMaxLayer();

            network.RandomizeWeights(0.1);

            return network;
        }

        private static void TrainModel(Network model, Func<Image, Array> getInput)
        {
            int total = 1200;
            int current = 0;
            using (var pbar = new ProgressBar(total, "Training Model"))
            {
                foreach (var image in ImageReader.ReadTrainingData(total))
                {
                    var target = Helper.CreateTarget(image.Label, NumberOfClasses);
                    model.TrainModel(getInput(image), target);
                    pbar.Tick($"Training Model: {++current} of {total}");
                }
            }
        }


        private static double TestModel(Model model, Func<Image, Array> getInput)
        {
            var results = new List<double>();
            int total = 600;
            int current = 0;
            using (var pbar = new ProgressBar(total, "Testing Model"))
            {
                foreach (var image in ImageReader.ReadTestData(total))
                {
                    var res = model.Predict(getInput(image));
                    results.Add(Helper.IntegerFromOutput(res) == image.Label ? 1 : 0);
                    pbar.Tick($"Testing Model: {++current} of {total}");
                }
            }

            return results.Average();
        }
    }
}
