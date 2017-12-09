using Ann.Activators;
using Ann.Configuration;
using Ann.Connections;
using Ann.Layers;
using Ann.Neurons;
using Ann.Resources;
using Ann.WeightInitializers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Ann
{
    public class Network
    {
        private readonly List<Layer> _layers;
        private readonly NetworkConfiguration _config;

        public Network(NetworkConfiguration config)
        {
            _config = config;
            _layers = CreateLayers(config.LayerConfiguration);

            ValidateNetworkConfiguration();
        }


        /// <summary>
        /// Process set of training data using gradient descent with back propagation alghoritm
        /// </summary>
        /// <param name="input">Input parametres</param>
        /// <param name="target">Target output</param>
        /// <returns>Error</returns>
        public double Train(List<double> input, List<double> target)
        {
            ValidateTrainingData(input, target);
            GetInputLayer().SetInputs(input);
            GetOutputLayer().SetTarget(target);

            ForwardPass();
            BackwardPass();
            return GetOutputLayer().GetTotalError();
        }

        public List<Layer> GetLayers()
        {
            return _layers;
        }

        #region helpers
        private void ForwardPass()
        {
            foreach (var layer in _layers.Where(q => !(q is InputLayer)))
            {
                layer.CalculateValue();
            }
        }
        private void BackwardPass()
        {
            foreach (var layer in _layers.FastReverse())
            {
                layer.CalculateError();
            }
        }
        private void RandomizeWeights()
        {
            foreach (var layer in _layers.Where(q => !(q is InputLayer)))
            {
                layer.RandomizeWeights();
            }
        }
        private void ValidateTrainingData(IEnumerable<double> input, IEnumerable<double> target)
        {
            if (input.Count() != _layers.First().Neurons.Count)
            {
                throw new Exception(Messages.InvalidInputArguments);
            }

            if (target.Count() != _layers.Last().Neurons.Count)
            {
                throw new Exception(Messages.InvalidTargetOutput);
            }
        }
        private void ValidateNetworkConfiguration()
        {
            if(_layers.OfType<InputLayer>().Count() != 1)
            {
                throw new Exception(Messages.InvalidNumberOfInputLayers);
            }
            else if(_layers.OfType<OutputLayer>().Count() != 1)
            {
                throw new Exception(Messages.InvalidNumberOfOutputLayers);
            }
            else if(!_layers.OfType<Layer>().Any())
            {
                throw new Exception(Messages.InvalidNumberOfHiddenLayers);
            }
        }
        private InputLayer GetInputLayer() => _layers.OfType<InputLayer>().Single();
        private OutputLayer GetOutputLayer() => _layers.OfType<OutputLayer>().Single();
        private List<Connection> CreateConnections(int prevLayerIndex, int nextLayerIndex)
        {
            var prev = _config.LayerConfiguration.Layers[prevLayerIndex].NumberOfNeurons;
            var next = _config.LayerConfiguration.Layers[nextLayerIndex].NumberOfNeurons;

            List<Connection> cList = new List<Connection>();
            for (int j = 1; j < prev; j++)
            {
                for (int k = 1; k < next; k++)
                {
                    cList.Add(new Connection(new Coordinate(j, prevLayerIndex), new Coordinate(k, nextLayerIndex)));
                }
            }

            return cList;
        }
        private List<Layer> CreateLayers(LayerConfiguration config)
        {
            List<Layer> layers = new List<Layer>();

            for (int i = 0; i < config.Layers.Count; i++)
            {
                var cur = config.Layers[i];

                if (i == 0)
                {
                    List<InputNeuron> nList = new List<InputNeuron>();
                    for (int j = 1; j <= cur.NumberOfNeurons; j++)
                    {
                        nList.Add(new InputNeuron(j));
                    }

                    layers.Add(new InputLayer(nList));
                }
                else if (i != config.Layers.Count)
                {
                    List<Neuron> nList = new List<Neuron>();
                    for (int j = 1; j <= cur.NumberOfNeurons; j++)
                    {
                        nList.Add(new Neuron(j, cur.Activator, cur.WeightInitializer));
                    }

                    layers.Add(new Layer(nList, i));
                }
                else if (i == config.Layers.Count - 1)
                {
                    List<OutputNeuron> nList = new List<OutputNeuron>();
                    for (int j = 1; j <= cur.NumberOfNeurons; j++)
                    {
                        nList.Add(new OutputNeuron(j, cur.Activator, cur.WeightInitializer));
                    }

                    layers.Add(new OutputLayer(nList, i));
                }
            }

            return CreateConnections(layers);
        }
        private List<Layer> CreateConnections(List<Layer> layers)
        {
            for (int i = 0; i < layers.Count; i++)
            {
                if (i == 0)
                {
                    continue;
                }
                else
                {
                    List<Connection> cList = CreateConnections(i - 1, i);

                    foreach (var neuron in layers[i - 1].Neurons)
                    {
                        List<ForwardConnection> fc = cList
                            .Where(q => q.Prev.LayerIndex == i - 1 && q.Prev.NeuronIndex == neuron.NeuronIndex)
                            .Select(q => new ForwardConnection(q,
                                layers[i]
                                .Neurons
                                .Single(w => w.NeuronIndex == q.Next.NeuronIndex && q.Next.LayerIndex == i)))
                            .ToList();

                        neuron.SetForwardConnections(fc);
                    }

                    foreach (var neuron in layers[i].Neurons)
                    {
                        List<BackwardConnection> bc = cList
                            .Where(q => q.Next.LayerIndex == i && q.Next.NeuronIndex == neuron.NeuronIndex)
                            .Select(q => new BackwardConnection(q,
                                layers[i - 1]
                                .Neurons
                                .Single(w => w.NeuronIndex == q.Prev.NeuronIndex && q.Prev.LayerIndex == i - 1)))
                            .ToList();

                        neuron.SetBackwardConnections(bc);
                    }
                }
            }

            return layers;
        }
        #endregion
    }
}
