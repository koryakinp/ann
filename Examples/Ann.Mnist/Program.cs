﻿using System;
using System.Collections.Generic;
using System.Linq;
using Ann.Activators;
using Ann.LossFunctions;
using Gdo.Optimizers;
using ShellProgressBar;

namespace Ann.Mnist
{
    class Program
    {
        private static readonly int NumberOfClasses = 10;

        static void Main(string[] args)
        {
            var network = CreateModel();
            TrainModel(network, q => Helper.Create3DInput(q.Data));
            var model = network.BuildModel();
            var ratio = TestModel(model, q => Helper.Create3DInput(q.Data));
            Console.WriteLine($"Accuracy: {ratio * 100}% ");
            Console.ReadLine();
        }

        private static Network CreateModel()
        {
            var network = new Network(LossFunctionType.CrossEntropy, new Flat(0.001), NumberOfClasses);

            network.AddInputLayer(28, 1);
            network.AddConvolutionLayer(16, 5);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddPoolingLayer(2);
            network.AddConvolutionLayer(32, 5);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddPoolingLayer(2);
            network.AddFlattenLayer();
            network.AddDenseLayer(1024, true);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddDenseLayer(NumberOfClasses, false);
            network.AddSoftMaxLayer();

            network.RandomizeWeights(0.1);

            return network;
        }

        private static void TrainModel(Network model, Func<Image,Array> getInput)
        {
            int total = 10000;
            int current = 0;
            using (var pbar = new ProgressBar(total, "Training Model"))
            {
                foreach (var image in MnistReader.ReadTrainingData(total))
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
            int total = 1000;
            int current = 0;
            using (var pbar = new ProgressBar(total, "Testing Model"))
            {
                foreach (var image in MnistReader.ReadTestData(total))
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
