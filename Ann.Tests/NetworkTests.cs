using Ann.Activators;
using Ann.Configuration;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Xunit;

namespace Ann.Tests
{
    public class NetworkTests
    {
        [Fact]
        public void ShouldLoadNetworkConfiguration()
        {
            string filepath = "network-configuration.json";
            if (File.Exists(filepath))
            {
                File.Delete(filepath);
            }

            LayerConfiguration lc = new LayerConfiguration()
                .AddInputLayer(5)
                .AddHiddenLayer(10, ActivatorType.LogisticActivator)
                .AddHiddenLayer(11, ActivatorType.ReluActivator)
                .AddHiddenLayer(12, ActivatorType.TanhActivator)
                .AddOutputLayer(8, ActivatorType.LogisticActivator);

            NetworkConfiguration nc = new NetworkConfiguration(lc);

            Network network = new Network(nc);
            network.SaveModelToJson(filepath);

            Network network2 = new Network(filepath);

            Assert.Equal(network2._layers.Count, network._layers.Count);

            for (int i = 0; i < network._layers.Count; i++)
            {
                Assert.Equal(
                    network2._layers[i].Neurons.Count, 
                    network._layers[i].Neurons.Count);
            }

            for (int i = 0; i < network._layers.Count; i++)
            {
                for (int j = 0; j < network._layers[i].Neurons.Count; j++)
                {
                    Assert.Equal(
                        network2._layers[i].Neurons[j].NeuronIndex, 
                        network._layers[i].Neurons[j].NeuronIndex);
                }
            }


            Assert.Equal(network2._layers[0].LayerIndex, 1);
            Assert.Equal(network2._layers[1].LayerIndex, 2);
            Assert.Equal(network2._layers[2].LayerIndex, 3);
            Assert.Equal(network2._layers[3].LayerIndex, 4);
            Assert.Equal(network2._layers[4].LayerIndex, 5);

            Assert.Null(network2._layers[0].Neurons.First().Activator);
            Assert.IsType(typeof(LogisticActivator), network2._layers[1].Neurons.First().Activator);
            Assert.IsType(typeof(ReluActivator), network2._layers[2].Neurons.First().Activator);
            Assert.IsType(typeof(TanhActivator), network2._layers[3].Neurons.First().Activator);
            Assert.IsType(typeof(LogisticActivator), network2._layers[4].Neurons.First().Activator);
        }

        [Fact]
        public void ShouldLoadNetworkConfigurationWeithWeights()
        {
            Network network = new Network("configuration.json");

            CheckConnectionWeightsForNeuron(network, 1, 1, new List<double> { -0.06, 0.17, -0.44 });
            CheckConnectionWeightsForNeuron(network, 1, 2, new List<double> { 0.11, 0.04, 0.37 });

            CheckConnectionWeightsForNeuron(network, 2, 1, new List<double> { 0.88, 0.76, -0.66 });
            CheckConnectionWeightsForNeuron(network, 2, 2, new List<double> { 0.54, -0.66, -0.22 });
            CheckConnectionWeightsForNeuron(network, 2, 3, new List<double> { 0.53, 0.53, 0.50 });

            CheckConnectionWeightsForNeuron(network, 3, 1, new List<double> { -0.36, -0.75, 0.98 });
            CheckConnectionWeightsForNeuron(network, 3, 2, new List<double> { -0.85, 0.41, -0.26 });
            CheckConnectionWeightsForNeuron(network, 3, 3, new List<double> { -0.70, -0.40, 0.40 });

            CheckConnectionWeightsForNeuron(network, 4, 1, new List<double> { -0.98, -0.33, -0.17 });
            CheckConnectionWeightsForNeuron(network, 4, 2, new List<double> { -0.13, -0.18, 0.26 });
            CheckConnectionWeightsForNeuron(network, 4, 3, new List<double> { 0.53, 0.55, -0.91 });

            CheckConnectionWeightsForNeuron(network, 5, 1, new List<double> { 0.46, 1.00 });
            CheckConnectionWeightsForNeuron(network, 5, 2, new List<double> { -0.10, 0.54 });
            CheckConnectionWeightsForNeuron(network, 5, 3, new List<double> { -0.47, 0.34 });
        }


        [Fact]
        public void ShouldLoadNetworkConfigurationWeithBiases()
        {
            Network network = new Network("configuration.json");

            CheckBiasForNeuron(network, 1, 1, 0);
            CheckBiasForNeuron(network, 1, 2, 0);

            CheckBiasForNeuron(network, 2, 1, 0.84);
            CheckBiasForNeuron(network, 2, 2, 0.99);
            CheckBiasForNeuron(network, 2, 3, 0.69);

            CheckBiasForNeuron(network, 3, 1, -0.10);
            CheckBiasForNeuron(network, 3, 2, -0.32);
            CheckBiasForNeuron(network, 3, 3, -0.23);

            CheckBiasForNeuron(network, 4, 1, 0.51);
            CheckBiasForNeuron(network, 4, 2, -0.89);
            CheckBiasForNeuron(network, 4, 3, 0.37);

            CheckBiasForNeuron(network, 5, 1, -0.92);
            CheckBiasForNeuron(network, 5, 2, 0.94);
            CheckBiasForNeuron(network, 5, 3, -0.12);

            CheckBiasForNeuron(network, 6, 1, 0.63);
            CheckBiasForNeuron(network, 6, 2, -0.05);
        }

        private void CheckConnectionWeightsForNeuron(Network network, int layer, int neuron, List<double> weights)
        {
            var connections = network
                ._layers
                .First(q => q.LayerIndex == layer)
                .Neurons
                .First(q => q.NeuronIndex == neuron)
                .ForwardConnections;

            for (int i = 0; i < connections.Count; i++)
            {
                Assert.Equal(weights[i], connections[i].GetWeight());
            }
        }

        private void CheckBiasForNeuron(Network network, int layer, int neuron, double value)
        {
            var bias = network
                ._layers
                .First(q => q.LayerIndex == layer)
                .Neurons
                .First(q => q.NeuronIndex == neuron)
                .Bias;

            Assert.Equal(value, bias);
        }
    }
}
