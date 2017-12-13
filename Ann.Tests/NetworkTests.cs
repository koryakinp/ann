using Ann.Activators;
using Ann.Configuration;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
        public void ShouldLoadNetworkConfigurationWithWeights()
        {
            Network network = new Network("configuration.json");

            CheckConnectionWeightsForNeuron(network, 1, 1, new List<double> { -0.06, 0.17, -0.44 }, 10);
            CheckConnectionWeightsForNeuron(network, 1, 2, new List<double> { 0.11, 0.04, 0.37 }, 10);

            CheckConnectionWeightsForNeuron(network, 2, 1, new List<double> { 0.88, 0.76, -0.66 }, 10);
            CheckConnectionWeightsForNeuron(network, 2, 2, new List<double> { 0.54, -0.66, -0.22 }, 10);
            CheckConnectionWeightsForNeuron(network, 2, 3, new List<double> { 0.53, -0.33, 0.50 }, 10);

            CheckConnectionWeightsForNeuron(network, 3, 1, new List<double> { -0.36, -0.75, 0.98 }, 10);
            CheckConnectionWeightsForNeuron(network, 3, 2, new List<double> { -0.85, 0.41, -0.26 }, 10);
            CheckConnectionWeightsForNeuron(network, 3, 3, new List<double> { -0.70, -0.40, 0.40 }, 10);

            CheckConnectionWeightsForNeuron(network, 4, 1, new List<double> { -0.98, -0.33, -0.17 }, 10);
            CheckConnectionWeightsForNeuron(network, 4, 2, new List<double> { -0.13, -0.18, 0.26 }, 10);
            CheckConnectionWeightsForNeuron(network, 4, 3, new List<double> { 0.53, 0.55, -0.91 }, 10);

            CheckConnectionWeightsForNeuron(network, 5, 1, new List<double> { 0.46, 1.00 }, 10);
            CheckConnectionWeightsForNeuron(network, 5, 2, new List<double> { -0.10, 0.54 }, 10);
            CheckConnectionWeightsForNeuron(network, 5, 3, new List<double> { -0.47, 0.34 }, 10);
        }

        [Fact]
        public void ShouldLoadNetworkConfigurationWithBiases()
        {
            Network network = new Network("configuration.json");

            CheckBiasForNeuron(network, 1, 1, 0, 10);
            CheckBiasForNeuron(network, 1, 2, 0, 10);

            CheckBiasForNeuron(network, 2, 1, 0.84, 10);
            CheckBiasForNeuron(network, 2, 2, 0.99, 10);
            CheckBiasForNeuron(network, 2, 3, 0.69, 10);

            CheckBiasForNeuron(network, 3, 1, -0.10, 10);
            CheckBiasForNeuron(network, 3, 2, -0.32, 10);
            CheckBiasForNeuron(network, 3, 3, -0.23, 10);

            CheckBiasForNeuron(network, 4, 1, 0.51, 10);
            CheckBiasForNeuron(network, 4, 2, -0.89, 10);
            CheckBiasForNeuron(network, 4, 3, 0.37, 10);

            CheckBiasForNeuron(network, 5, 1, -0.92, 10);
            CheckBiasForNeuron(network, 5, 2, 0.94, 10);
            CheckBiasForNeuron(network, 5, 3, -0.12, 10);

            CheckBiasForNeuron(network, 6, 1, 0.63, 10);
            CheckBiasForNeuron(network, 6, 2, -0.05, 10);
        }

        [Fact]
        public void ForwardPassTest()
        {
            Network network = new Network("configuration.json");

            var error = network.TrainModel(new List<double> { 0.71, 0.65 }, new List<double> { 1, 0 });
            Assert.Equal(0.2979502329 , Math.Round(error, 10));
        }

        [Fact]
        public void ShouldUpdateDeltasAfterBackwardPass()
        {
            Network network = new Network("configuration.json");

            var error = network.TrainModel(new List<double> { 0.71, 0.65 }, new List<double> { 1, 0 });

            CheckDeltaForNeuron(network, 6, 1, -0.0856240032, 10);
            CheckDeltaForNeuron(network, 6, 2, 0.1480058634, 10);

            CheckDeltaForNeuron(network, 5, 1, 0.0219080207, 10);
            CheckDeltaForNeuron(network, 5, 2, 0.0158156959, 10);
            CheckDeltaForNeuron(network, 5, 3, 0.0190853930, 10);

            CheckDeltaForNeuron(network, 4, 1, -0.0072579064, 10);
            CheckDeltaForNeuron(network, 4, 2, -0.0001114570, 10);
            CheckDeltaForNeuron(network, 4, 3, 0.0005276072, 10);

            CheckDeltaForNeuron(network, 3, 1, 0.0005491146, 10);
            CheckDeltaForNeuron(network, 3, 2, 0.0014072728, 10);
            CheckDeltaForNeuron(network, 3, 3, 0.0012424223, 10);

            CheckDeltaForNeuron(network, 2, 1, 0.0001525386, 10);
            CheckDeltaForNeuron(network, 2, 2, -0.0001665540, 10);
            CheckDeltaForNeuron(network, 2, 3, 0.0001019127, 10);

            CheckDeltaForNeuron(network, 1, 1, 0, 10);
            CheckDeltaForNeuron(network, 1, 2, 0, 10);
        }

        [Fact]
        public void ShouldUpdateWeightsAfterBackwardPass()
        {
            Network network = new Network("configuration.json");

            network.TrainModel(new List<double> { 0.71, 0.65 }, new List<double> { 1, 0 });

            CheckConnectionWeightsForNeuron(network, 5, 1, new List<double> { 0.4623993481, 0.9958525930 }, 10);
            CheckConnectionWeightsForNeuron(network, 5, 2, new List<double> { -0.0934330652, 0.5286486870 }, 10);
            CheckConnectionWeightsForNeuron(network, 5, 3, new List<double> { -0.4674154450, 0.3355324526 }, 10);

            CheckConnectionWeightsForNeuron(network, 4, 1, new List<double> { -0.9809052579, -0.3306535179, -0.1707886245 }, 10);
            CheckConnectionWeightsForNeuron(network, 4, 2, new List<double> { -0.1304100099, -0.1802959917, 0.2596428157 }, 10);
            CheckConnectionWeightsForNeuron(network, 4, 3, new List<double> { 0.5283221813, 0.5487887600, -0.9114616486 }, 10);

            CheckConnectionWeightsForNeuron(network, 3, 1, new List<double> { -0.3594329494, -0.7499912920, 0.9799587787 }, 10);
            CheckConnectionWeightsForNeuron(network, 3, 2, new List<double> { -0.8497257568, 0.4100042115, -0.2600199359 }, 10);
            CheckConnectionWeightsForNeuron(network, 3, 3, new List<double> { -0.6997322060, -0.3999958876, 0.3999805330 }, 10);

            CheckConnectionWeightsForNeuron(network, 2, 1, new List<double> { 0.8799613140, 0.7599008553, -0.6600875307 }, 10);
            CheckConnectionWeightsForNeuron(network, 2, 2, new List<double> { 0.5399584280, -0.6601065409, -0.2200940605 }, 10);
            CheckConnectionWeightsForNeuron(network, 2, 3, new List<double> { 0.5299643193, -0.3300914427, 0.4999192691 }, 10);

            CheckConnectionWeightsForNeuron(network, 1, 1, new List<double> { -0.0600108302, 0.1700118253, -0.4400072358 }, 10);
            CheckConnectionWeightsForNeuron(network, 1, 2, new List<double> { 0.1099900850, 0.0400108260, 0.3699933757 }, 10);
        }

        [Fact]
        public void ShouldUpdateBiasesAfterBackwardPass()
        {
            Network network = new Network("configuration.json");

            network.TrainModel(new List<double> { 0.71, 0.65 }, new List<double> { 1, 0 });

            CheckBiasForNeuron(network, 6, 1, 0.6385624003, 10);
            CheckBiasForNeuron(network, 6, 2, -0.0648005863, 10);

            CheckBiasForNeuron(network, 5, 1, -0.9221908021, 10);
            CheckBiasForNeuron(network, 5, 2, 0.9384184304, 10);
            CheckBiasForNeuron(network, 5, 3, -0.1219085393, 10);

            CheckBiasForNeuron(network, 4, 1, 0.5107257906, 10);
            CheckBiasForNeuron(network, 4, 2, -0.8899888543, 10);
            CheckBiasForNeuron(network, 4, 3, 0.3699472393, 10);

            CheckBiasForNeuron(network, 3, 1, -0.1000549115, 10);
            CheckBiasForNeuron(network, 3, 2, -0.3201407273, 10);
            CheckBiasForNeuron(network, 3, 3, -0.2301242422, 10);

            CheckBiasForNeuron(network, 2, 1, 0.8399847461, 10);
            CheckBiasForNeuron(network, 2, 2, 0.9900166554, 10);
            CheckBiasForNeuron(network, 2, 3, 0.6899898087, 10);

            CheckBiasForNeuron(network, 1, 1, 0 ,10);
            CheckBiasForNeuron(network, 1, 2, 0, 10);
        }

        [Fact]
        public void ShouldReduceTotalError()
        {
            Network network = new Network("configuration.json");

            var error1 = network.TrainModel(new List<double> { 0.71, 0.65 }, new List<double> { 1, 0 });
            var error2 = network.TrainModel(new List<double> { 0.71, 0.65 }, new List<double> { 1, 0 });
            var error3 = network.TrainModel(new List<double> { 0.71, 0.65 }, new List<double> { 1, 0 });

            Assert.Equal(Math.Round(error1, 10), Math.Round(0.2979502329, 10));
            Assert.Equal(Math.Round(error2, 10), Math.Round(0.2926133141, 10));
            Assert.Equal(Math.Round(error3, 10), Math.Round(0.2873160054, 10));
        }

        private void CheckConnectionWeightsForNeuron(Network network, int layer, int neuron, List<double> weights, int precission)
        {
            var connections = network
                ._layers
                .First(q => q.LayerIndex == layer)
                .Neurons
                .First(q => q.NeuronIndex == neuron)
                .ForwardConnections;

            for (int i = 0; i < connections.Count; i++)
            {
                
                Assert.Equal(Math.Round(weights[i], precission), Math.Round(connections[i].GetWeight(), precission));
            }
        }

        private void CheckBiasForNeuron(Network network, int layer, int neuron, double value, int decimals)
        {
            var bias = network
                ._layers
                .First(q => q.LayerIndex == layer)
                .Neurons
                .First(q => q.NeuronIndex == neuron)
                .Bias;

            Assert.Equal(Math.Round(value, 10), Math.Round(bias, 10));
        }

        private void CheckDeltaForNeuron(Network network, int layer, int neuron, double value, int decimals)
        {
            var delta = network
                ._layers
                .First(q => q.LayerIndex == layer)
                .Neurons
                .First(q => q.NeuronIndex == neuron)
                .Delta;

            Assert.Equal(Math.Round(value, decimals), Math.Round(delta, decimals));
        }
    }
}
