using Ann.Configuration;
using System;
using System.Collections.Generic;

namespace Ann.Client
{
    class Program
    {
        static void Main(string[] args)
        {
            Random rnd = new Random();

            var layerConfig = new LayerConfiguration();

            layerConfig
                .AddInputLayer(2)
                .AddHiddenLayer(3, ActivatorType.LogisticActivator)
                .AddHiddenLayer(3, ActivatorType.LogisticActivator)
                .AddHiddenLayer(3, ActivatorType.LogisticActivator)
                .AddOutputLayer(1, ActivatorType.LogisticActivator);

            var config = new NetworkConfiguration(layerConfig, 0.1, 0.95);

            var network = new Network(config);

            for (int i = 0; i < 1000; i++)
            {
                bool i1 = rnd.NextDouble() >= 0.5;
                bool i2 = rnd.NextDouble() >= 0.5;

                bool o1 = i1 ^ i2;

                var error = network.TrainModel(new List<double> { i1 ? 1 : 0, i2 ? 1 : 0 }, new List<double> { o1 ? 1 : 0 });
                Console.WriteLine(error);
                Console.ReadKey();
            }
        }
    }
}
