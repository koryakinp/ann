using Ann.Activators;
using Ann.Configuration;
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


            Assert.Equal(1, network2._layers[0].LayerIndex);
            Assert.Equal(2, network2._layers[1].LayerIndex);
            Assert.Equal(3, network2._layers[2].LayerIndex);
            Assert.Equal(4, network2._layers[3].LayerIndex);
            Assert.Equal(5, network2._layers[4].LayerIndex);

            Assert.Null(network2._layers[0].Neurons.First().Activator);
            Assert.IsType<LogisticActivator>(network2._layers[1].Neurons.First().Activator);
            Assert.IsType<ReluActivator>(network2._layers[2].Neurons.First().Activator);
            Assert.IsType<TanhActivator>(network2._layers[3].Neurons.First().Activator);
            Assert.IsType<LogisticActivator>(network2._layers[4].Neurons.First().Activator);
        }

        [Fact]
        public void ShouldLoadNetworkConfigurationWithWeights()
        {
            Network network = new Network("configuration.json");

            CheckConnectionWeightsForNeuron(network, 1, 1, new List<double> { -0.06, 0.17, -0.44 });
            CheckConnectionWeightsForNeuron(network, 1, 2, new List<double> { 0.11, 0.04, 0.37 });

            CheckConnectionWeightsForNeuron(network, 2, 1, new List<double> { 0.88, 0.76, -0.66 });
            CheckConnectionWeightsForNeuron(network, 2, 2, new List<double> { 0.54, -0.66, -0.22 });
            CheckConnectionWeightsForNeuron(network, 2, 3, new List<double> { 0.53, -0.33, 0.50 });

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
        public void ShouldLoadNetworkConfigurationWithBiases()
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

        [Fact]
        public void ForwardPassTest()
        {
            Network network = new Network("configuration.json");

            var error = network.TrainModel(new List<double> { 0.71, 0.65 }, new List<double> { 1, 0 });
            Assert.Equal(0.2979502329, error, 10);
        }

        [Fact]
        public void ShouldUpdateDeltasAfterBackwardPass()
        {
            Network network = new Network("configuration.json");

            var error = network.TrainModel(new List<double> { 0.71, 0.65 }, new List<double> { 1, 0 });

            CheckDeltaForNeuron(network, 6, 1, -0.0856240032);
            CheckDeltaForNeuron(network, 6, 2, 0.1480058634);

            CheckDeltaForNeuron(network, 5, 1, 0.0219080207);
            CheckDeltaForNeuron(network, 5, 2, 0.0158156959);
            CheckDeltaForNeuron(network, 5, 3, 0.0190853930);

            CheckDeltaForNeuron(network, 4, 1, -0.0072579064);
            CheckDeltaForNeuron(network, 4, 2, -0.0001114570);
            CheckDeltaForNeuron(network, 4, 3, 0.0005276072);

            CheckDeltaForNeuron(network, 3, 1, 0.0005491146);
            CheckDeltaForNeuron(network, 3, 2, 0.0014072728);
            CheckDeltaForNeuron(network, 3, 3, 0.0012424223);

            CheckDeltaForNeuron(network, 2, 1, 0.0001525386);
            CheckDeltaForNeuron(network, 2, 2, -0.0001665540);
            CheckDeltaForNeuron(network, 2, 3, 0.0001019127);

            CheckDeltaForNeuron(network, 1, 1, 0);
            CheckDeltaForNeuron(network, 1, 2, 0);
        }

        [Fact]
        public void ShouldUpdateWeightsAfterBackwardPass()
        {
            Network network = new Network("configuration.json");

            network.TrainModel(new List<double> { 0.71, 0.65 }, new List<double> { 1, 0 });

            CheckConnectionWeightsForNeuron(network, 5, 1, new List<double> { 0.4623993481, 0.9958525930 });
            CheckConnectionWeightsForNeuron(network, 5, 2, new List<double> { -0.0934330652, 0.5286486870 });
            CheckConnectionWeightsForNeuron(network, 5, 3, new List<double> { -0.4674154450, 0.3355324526 });

            CheckConnectionWeightsForNeuron(network, 4, 1, new List<double> { -0.9809052579, -0.3306535179, -0.1707886245 });
            CheckConnectionWeightsForNeuron(network, 4, 2, new List<double> { -0.1304100099, -0.1802959917, 0.2596428157 });
            CheckConnectionWeightsForNeuron(network, 4, 3, new List<double> { 0.5283221813, 0.5487887600, -0.9114616486 });

            CheckConnectionWeightsForNeuron(network, 3, 1, new List<double> { -0.3594329494, -0.7499912920, 0.9799587787 });
            CheckConnectionWeightsForNeuron(network, 3, 2, new List<double> { -0.8497257568, 0.4100042115, -0.2600199359 });
            CheckConnectionWeightsForNeuron(network, 3, 3, new List<double> { -0.6997322060, -0.3999958876, 0.3999805330 });

            CheckConnectionWeightsForNeuron(network, 2, 1, new List<double> { 0.8799613140, 0.7599008553, -0.6600875307 });
            CheckConnectionWeightsForNeuron(network, 2, 2, new List<double> { 0.5399584280, -0.6601065409, -0.2200940605 });
            CheckConnectionWeightsForNeuron(network, 2, 3, new List<double> { 0.5299643193, -0.3300914427, 0.4999192691 });

            CheckConnectionWeightsForNeuron(network, 1, 1, new List<double> { -0.0600108302, 0.1700118253, -0.4400072358 });
            CheckConnectionWeightsForNeuron(network, 1, 2, new List<double> { 0.1099900850, 0.0400108260, 0.3699933757 });
        }

        [Fact]
        public void ShouldUpdateBiasesAfterBackwardPass()
        {
            Network network = new Network("configuration.json");

            network.TrainModel(new List<double> { 0.71, 0.65 }, new List<double> { 1, 0 });

            CheckBiasForNeuron(network, 6, 1, 0.6385624003);
            CheckBiasForNeuron(network, 6, 2, -0.0648005863);

            CheckBiasForNeuron(network, 5, 1, -0.9221908021);
            CheckBiasForNeuron(network, 5, 2, 0.9384184304);
            CheckBiasForNeuron(network, 5, 3, -0.1219085393);

            CheckBiasForNeuron(network, 4, 1, 0.5107257906);
            CheckBiasForNeuron(network, 4, 2, -0.8899888543);
            CheckBiasForNeuron(network, 4, 3, 0.3699472393);

            CheckBiasForNeuron(network, 3, 1, -0.1000549115);
            CheckBiasForNeuron(network, 3, 2, -0.3201407273);
            CheckBiasForNeuron(network, 3, 3, -0.2301242422);

            CheckBiasForNeuron(network, 2, 1, 0.8399847461);
            CheckBiasForNeuron(network, 2, 2, 0.9900166554);
            CheckBiasForNeuron(network, 2, 3, 0.6899898087);

            CheckBiasForNeuron(network, 1, 1, 0);
            CheckBiasForNeuron(network, 1, 2, 0);
        }

        [Fact]
        public void ShouldReduceTotalError()
        {
            Network network = new Network("configuration.json");

            var error1 = network.TrainModel(new List<double> { 0.71, 0.65 }, new List<double> { 1, 0 });
            var error2 = network.TrainModel(new List<double> { 0.71, 0.65 }, new List<double> { 1, 0 });
            var error3 = network.TrainModel(new List<double> { 0.71, 0.65 }, new List<double> { 1, 0 });

            Assert.Equal(0.2979502329, error1, 10);
            Assert.Equal(0.2926133141, error2, 10);
            Assert.Equal(0.2873160054, error3, 10);
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
                Assert.Equal(weights[i], connections[i].GetWeight(), 10);
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

            Assert.Equal(value, bias, 10);
        }

        private void CheckDeltaForNeuron(Network network, int layer, int neuron, double value)
        {
            var delta = network
                ._layers
                .First(q => q.LayerIndex == layer)
                .Neurons
                .First(q => q.NeuronIndex == neuron)
                .Delta;

            Assert.Equal(value, delta, 10);
        }
    }
}
