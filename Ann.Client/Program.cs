using Ann.Configuration;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ImageSharp;
using Ann.Decayers;
using Ann.Utils;
using Image = Ann.Utils.Image;

namespace Ann.Client
{
    class Program
    {
        private static MovingAverage ma = new MovingAverage(100);

        static void Main(string[] args)
        {
            List<Image> images = ImageProvider.ProvideImages(@"minst/train.zip");

            var layerConfig = new LayerConfiguration();

            layerConfig
                .AddInputLayer(784)
                .AddHiddenLayer(16, ActivatorType.LogisticActivator)
                .AddHiddenLayer(16, ActivatorType.LogisticActivator)
                .AddOutputLayer(10, ActivatorType.LogisticActivator);

            var config = new NetworkConfiguration(layerConfig)
            {
                Momentum = 0.9,
                LearningRate = 0.1,
                LearningRateDecayer = new StepDecayer(0.1, 0.8, 10000)
            };

            var network = new Network(config);

            foreach (var image in images)
            {
                List<double> data = image.Data.Select(q => (double)q / 255).ToList();
                List<double> target = NetworkHelper.CreateTarget(image.Value);

                var error = network.TrainModel(data, target);
                Console.WriteLine(ma.Compute(error).ToString("#.##"));
            }

            network.SaveModelToJson("network-model.json");

            List<Image> testImages = ImageProvider.ProvideImages(@"minst/test.zip");

            int success = 0;
            int fail = 0;

            List<Image> succes = new List<Image>();
            List<Image> fails = new List<Image>();

            for (int i = 0; i < testImages.Count; i++)
            {
                var image = testImages[i];
                List<double> data = image.Data.Select(q => (double)q / 255).ToList();
                var res = network.UseModel(data);
                int predicted = NetworkHelper.IntegerFromOutput(res);
                if(predicted == image.Value)
                {
                    succes.Add(image);
                    success++;
                }
                else
                {
                    fails.Add(image);
                    fail++;
                }
            }




            Console.WriteLine($"Success: {success} | Fail: {fail}");

            Console.ReadKey();
        }
    }
}
