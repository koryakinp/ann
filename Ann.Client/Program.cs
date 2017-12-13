using Ann.Configuration;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ImageSharp;
using ImageIng = ImageSharp.Image;

namespace Ann.Client
{
    class Program
    {
        private static MovingAverage ma = new MovingAverage(100);

        static void Main(string[] args)
        {
            List<Image> images = CreateTraingImages();

            var layerConfig = new LayerConfiguration();

            layerConfig
                .AddInputLayer(784)
                .AddHiddenLayer(20, ActivatorType.LogisticActivator)
                .AddHiddenLayer(20, ActivatorType.LogisticActivator)
                .AddOutputLayer(10, ActivatorType.LogisticActivator);

            var config = new NetworkConfiguration(layerConfig, 0.10, 0.95);

            var network = new Network(config);

            foreach (var image in images)
            {
                var error = network.TrainModel(
                    image.Data.Select(q => (double)q/255).ToList(), 
                    CreateTarget(image.Value));

                Console.WriteLine(ma.Compute(error).ToString("#.##"));
            }

            List<Image> testImages = CreateTestImages();
            //List<Image> testImages = CreateCustomTestImages();

            for (int i = 0; i < testImages.Count; i++)
            {
                var image = testImages[i];
                var res = network.UseModel(image.Data.Select(q => (double)q / 255).ToList());
                image.Value = IntegerFromOutput(res);
            }

            foreach (var img in testImages.Take(100))
            {
                ImageParser.Parse(img);
            } 

            Console.ReadKey();
        }

        private static List<double> CreateTarget(byte value)
        {
            List<double> res = new List<double>();

            for (int i = 1; i <= 10; i++)
            {
                res.Add(i == value ? 1 : 0);
            }

            return res;
        }

        private static byte IntegerFromOutput (List<double> values)
        {
            return (byte)(values.IndexOf(values.Max()) + 1);
        }

        private static List<Image> CreateTraingImages()
        {
            List<Image> images = new List<Image>();

            using (var reader = new StreamReader(@"train.csv"))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    images.Add(new Image
                    {
                        Value = byte.Parse(line.Split(',').First()),
                        Data = line.Split(',').Skip(1).Select(q => byte.Parse(q)).ToList()
                    });
                }
            }

            return images;
        }

        private static List<Image> CreateTestImages()
        {
            List<Image> testImages = new List<Image>();

            using (var reader = new StreamReader(@"test.csv"))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    testImages.Add(new Image
                    {
                        Data = line.Split(',').Select(q => byte.Parse(q)).ToList()
                    });
                }
            }

            return testImages;
        }

        private static List<Image> CreateCustomTestImages()
        {
            List<Image> testImages = new List<Image>();

            foreach (var file in Directory.GetFiles("custom-images"))
            {
                testImages.Add(new Image
                {
                    Data = ImageParser.ConvertImageToBytes(file)
                });
            }

            return testImages;
        }
    }
}
