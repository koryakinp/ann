using Ann.Activators;
using Ann.Configuration;
using Ann.Connections;
using Ann.Layers;
using Ann.Model;
using Ann.Neurons;
using Ann.Resources;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Ann
{
    public class Network
    {
        internal readonly List<Layer> _layers;
        internal readonly NetworkConfiguration _config;
        private int _epoch = 1;
        private readonly double _initialLearningRate;

        /// <summary>
        /// Creates a Network objectd based on provided NetworkConfiguration
        /// </summary>
        /// <param name="config">Network configuration</param>
        public Network(NetworkConfiguration config)
        {
            _config = config;
            _layers = CreateLayers(config.LayerConfiguration);
            CreateConnections();
            ValidateNetworkConfiguration();
            RandomizeWeights();
            _initialLearningRate = _config.LearningRate;
        }

        public Network(string filepath)
        {
            NetworkModel model = Serializer.ReadFromJsonFile<NetworkModel>(filepath);
            LayerConfiguration layerConfiguration = CreateLayerConfiguration(model);
            _config = new NetworkConfiguration(layerConfiguration);
            _layers = CreateLayers(layerConfiguration);
            _initialLearningRate = _config.LearningRate;
            CreateConnections();
            AssignWeightValues(model);
            AssignBiasesValues(model);
            ValidateNetworkConfiguration();
        }

        /// <summary>
        /// Process an input data on a previously trained model
        /// </summary>
        /// <param name="inputs">Input values to be processed</param>
        /// <returns>Model output values</returns>
        public List<double> UseModel(List<double> inputs)
        {
            ValidateData(inputs);
            InputLayer.SetInputs(inputs);
            ForwardPass();
            return OutputLayer.GetOutputValues();
        }

        /// <summary>
        /// Process set of training data using gradient descent with back propagation alghoritm
        /// </summary>
        /// <param name="input">Input parametres</param>
        /// <param name="target">Target output</param>
        /// <returns>Error</returns>
        public double TrainModel(List<double> input, List<double> target)
        {
            ValidateTrainingData(input, target);
            InputLayer.SetInputs(input);
            OutputLayer.SetTarget(target);

            ForwardPass();
            BackwardPass();
            UpdateWeights();
            _epoch++;
            UpdateLearningRate();
            return OutputLayer.GetTotalError();
        }

        public void SaveModelToJson(string filename)
        {
            NetworkModel model = new NetworkModel();

            foreach (var layer in _layers)
            {
                model.Layers.Add(new LayerModel
                {
                    LayerIndex = layer.LayerIndex,
                    Neurons = layer.Neurons
                        .Select(q => q.ToNeuronModel())
                        .ToList()
                });
            }

            Serializer.WriteToJsonFile(filename, model);
        }

        #region helpers
        private void ForwardPass()
        {
            foreach (var layer in _layers
                .OrderBy(q => q.LayerIndex)
                .ToList())
            {
                layer.CalculateValue();
            }
        }
        private void BackwardPass()
        {
            foreach (var layer in _layers
                .Where(q => !(q is InputLayer))
                .OrderByDescending(q => q.LayerIndex)
                .ToList())
            {
                layer.CalculateError();
            }
        }
        private void UpdateWeights()
        {
            foreach (var layer in _layers
                .OrderByDescending(q => q.LayerIndex)
                .ToList())
            {
                foreach (var neuron in layer.Neurons)
                {
                    neuron.UpdateWeights(_config);
                }
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
        private void ValidateData(IEnumerable<double> input)
        {
            if (input.Count() != _layers.First().Neurons.Count)
            {
                throw new Exception(Messages.InvalidInputArguments);
            }
        }
        private void ValidateNetworkConfiguration()
        {
            if (_config.LearningRateDecayer == null)
            {
                if(_config.LearningRate < 0 || _config.LearningRate > 1)
                {
                    throw new Exception(Messages.InvalidLearningRate);
                }
            }

            if (_config.Momentum < 0 || _config.Momentum > 1)
            {
                throw new Exception(Messages.InvalidMomentum);
            }
            else if (_layers.OfType<InputLayer>().Count() != 1)
            {
                throw new Exception(Messages.InvalidNumberOfInputLayers);
            }
            else if (_layers.OfType<OutputLayer>().Count() != 1)
            {
                throw new Exception(Messages.InvalidNumberOfOutputLayers);
            }
            else if (!_layers.OfType<Layer>().Any())
            {
                throw new Exception(Messages.InvalidNumberOfHiddenLayers);
            }
        }
        private InputLayer InputLayer => _layers.OfType<InputLayer>().Single();
        private OutputLayer OutputLayer => _layers.OfType<OutputLayer>().Single();
        private List<Connection> CreateConnections(int prevLayerIndex, int nextLayerIndex)
        {
            var prev = _config.LayerConfiguration.Layers[prevLayerIndex].NumberOfNeurons;
            var next = _config.LayerConfiguration.Layers[nextLayerIndex].NumberOfNeurons;

            List<Connection> cList = new List<Connection>();
            for (int j = 0; j < prev; j++)
            {
                for (int k = 0; k < next; k++)
                {
                    cList.Add(new Connection(new Coordinate(j + 1, prevLayerIndex), new Coordinate(k + 1, nextLayerIndex)));
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
                else if (i != config.Layers.Count - 1)
                {
                    List<Neuron> nList = new List<Neuron>();
                    for (int j = 1; j <= cur.NumberOfNeurons; j++)
                    {
                        nList.Add(new Neuron(j, cur.Activator, cur.WeightInitializer));
                    }

                    layers.Add(new Layer(nList, i + 1));
                }
                else if (i == config.Layers.Count - 1)
                {
                    List<OutputNeuron> nList = new List<OutputNeuron>();
                    for (int j = 1; j <= cur.NumberOfNeurons; j++)
                    {
                        nList.Add(new OutputNeuron(j, cur.Activator, cur.WeightInitializer));
                    }

                    layers.Add(new OutputLayer(nList, i + 1));
                }
            }

            return layers;
        }
        private void CreateConnections()
        {
            for (int i = 1; i < _layers.Count; i++)
            {
                List<Connection> cList = CreateConnections(i - 1, i);

                foreach (var neuron in _layers[i - 1].Neurons)
                {
                    List<NeuronConnection> fc = cList
                        .Where(q => q.Prev.LayerIndex == i - 1 && q.Prev.NeuronIndex == neuron.NeuronIndex)
                        .Select(q => new NeuronConnection(q,
                            _layers[i]
                            .Neurons
                            .Single(w => w.NeuronIndex == q.Next.NeuronIndex && q.Next.LayerIndex == i)))
                        .ToList();

                    neuron.SetForwardConnections(fc);
                }

                foreach (var neuron in _layers[i].Neurons)
                {
                    List<NeuronConnection> bc = cList
                        .Where(q => q.Next.LayerIndex == i && q.Next.NeuronIndex == neuron.NeuronIndex)
                        .Select(q => new NeuronConnection(q,
                            _layers[i - 1]
                            .Neurons
                            .Single(w => w.NeuronIndex == q.Prev.NeuronIndex && q.Prev.LayerIndex == i - 1)))
                        .ToList();

                    neuron.SetBackwardConnections(bc);
                }
            }
        }

        private void AssignWeightValues(NetworkModel model)
        {
            foreach (var layer in _layers)
            {
                foreach (var neuron in layer.Neurons)
                {
                    var weights = model
                        .Layers
                        .First(q => q.LayerIndex == layer.LayerIndex)
                        .Neurons
                        .First(q => q.NeuronIndex == neuron.NeuronIndex)
                        .Weights;

                    for (int i = 0; i < neuron.ForwardConnections.Count; i++)
                    {
                        neuron.ForwardConnections[i].SetWeight(weights[i]);
                    }
                }
            }
        }

        private void AssignBiasesValues(NetworkModel model)
        {
            foreach (var layer in _layers)
            {
                foreach (var neuron in layer.Neurons)
                {
                    var bias = model
                        .Layers
                        .First(q => q.LayerIndex == layer.LayerIndex)
                        .Neurons
                        .First(q => q.NeuronIndex == neuron.NeuronIndex)
                        .Bias;

                    neuron.SetBias(bias);
                }
            }
        }

        private LayerConfiguration CreateLayerConfiguration(NetworkModel model)
        {
            LayerConfiguration layerConfiguration = new LayerConfiguration();

            for (int i = 0; i < model.Layers.Count; i++)
            {
                var layer = model.Layers[i];

                if (i == 0)
                {
                    layerConfiguration.AddInputLayer(layer.Neurons.Count);
                }
                else if (i != model.Layers.Count - 1)
                {
                    IActivator activator = (IActivator)Activator.CreateInstance(Type.GetType(layer.Neurons.First().Activator));
                    layerConfiguration.AddHiddenLayer(layer.Neurons.Count, activator);
                }
                else
                {
                    IActivator activator = (IActivator)Activator.CreateInstance(Type.GetType(layer.Neurons.First().Activator));
                    layerConfiguration.AddOutputLayer(layer.Neurons.Count, activator);
                }
            }

            return layerConfiguration;
        }

        private void UpdateLearningRate()
        {
            if(_config.LearningRateDecayer != null)
            {
                _config.LearningRate = _config.LearningRateDecayer.Decay(_epoch);
            }
        }
        #endregion
    }
}
