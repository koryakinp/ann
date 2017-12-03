using Ann.Activators;
using Ann.Layers;
using Ann.Neurons;
using Ann.Resources;
using Ann.WeightInitializers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Ann
{
    public class Network
    {
        private readonly List<Layer> _layers;
        private bool _saved;

        public Network()
        {
            _layers = new List<Layer>();
        }

        public Network AddInputLayer(int numberOfNeurons)
        {
            _layers.Add(new InputLayer(numberOfNeurons, null));
            return this;
        }

        #region AddHiddenLayer overloads
        public Network AddHiddenLayer(int numberOfNeurons)
        {
            _layers.Add(new Layer(numberOfNeurons, _layers.Count + 1, new TanhActivator(), new DefaultWeightInitializer()));
            return this;
        }

        public Network AddHiddenLayer(int numberOfNeurons, IActivator activator)
        {
            _layers.Add(new Layer(numberOfNeurons, _layers.Count + 1, activator, new DefaultWeightInitializer()));
            return this;
        }

        public Network AddHiddenLayer(int numberOfNeurons, ActivatorType type)
        {
            _layers.Add(new Layer(numberOfNeurons, _layers.Count + 1, ActivatorFactory.Produce(type), new DefaultWeightInitializer()));
            return this;
        }

        public Network AddHiddenLayer(int numberOfNeurons, IWeightInitializer weightInitializer)
        {
            _layers.Add(new Layer(numberOfNeurons, _layers.Count + 1, new TanhActivator(), weightInitializer));
            return this;
        }

        public Network AddHiddenLayer(int numberOfNeurons, IActivator activator, IWeightInitializer weightInitializer)
        {
            _layers.Add(new Layer(numberOfNeurons, _layers.Count + 1, activator, weightInitializer));
            return this;
        }

        public Network AddHiddenLayer(int numberOfNeurons, ActivatorType type, IWeightInitializer weightInitializer)
        {
            _layers.Add(new Layer(numberOfNeurons, _layers.Count + 1, ActivatorFactory.Produce(type), weightInitializer));
            return this;
        }
        #endregion

        #region AddOutputLayer overloads
        public Network AddOutputLayer(int numberOfNeurons)
        {
            _layers.Add(new OutputLayer(numberOfNeurons, _layers.Count + 1, new TanhActivator(), new DefaultWeightInitializer()));
            return this;
        }

        public Network AddOutputLayer(int numberOfNeurons, IActivator activator)
        {
            _layers.Add(new OutputLayer(numberOfNeurons, _layers.Count + 1, activator, new DefaultWeightInitializer()));
            return this;
        }

        public Network AddOutputLayer(int numberOfNeurons, ActivatorType type)
        {
            _layers.Add(new OutputLayer(numberOfNeurons, _layers.Count + 1, ActivatorFactory.Produce(type), new DefaultWeightInitializer()));
            return this;
        }

        public Network AddOutputLayer(int numberOfNeurons, IWeightInitializer weightInitializer)
        {
            _layers.Add(new OutputLayer(numberOfNeurons, _layers.Count + 1, new TanhActivator(), weightInitializer));
            return this;
        }

        public Network AddOutputLayer(int numberOfNeurons, IActivator activator, IWeightInitializer weightInitializer)
        {
            _layers.Add(new OutputLayer(numberOfNeurons, _layers.Count + 1, activator, weightInitializer));
            return this;
        }

        public Network AddOutputLayer(int numberOfNeurons, ActivatorType type, IWeightInitializer weightInitializer)
        {
            _layers.Add(new OutputLayer(numberOfNeurons, _layers.Count + 1, ActivatorFactory.Produce(type), weightInitializer));
            return this;
        }
        #endregion AddOutputLayer overloads

        public void Save()
        {
            ValidateNetworkConfiguration();
            GenerateConnections();
            RandomizeWeights();
            _saved = true;
        }

        public double Train(List<double> input, List<double> target)
        {
            ValidateTrainingData(input, target);

            GetInputLayer().SetInputs(input);
            GetOutputLayer().SetTarget(target);

            foreach (var layer in _layers)
            {
                if (layer is InputLayer) continue;

                layer.CalculateValue();
            }

            foreach (var layer in _layers.FastReverse())
            {
                layer.CalculateError();
            }

            return GetOutputLayer().GetTotalError();
        }

        #region helpers
        private void RandomizeWeights()
        {
            foreach (var layer in _layers.Where(q => !(q is InputLayer)))
            {
                layer.RandomizeWeights();
            }
        }
        private void ValidateTrainingData(IEnumerable<double> input, IEnumerable<double> target)
        {
            if (!_saved)
            {
                throw new Exception(Messages.NetworkNotSaved);
            }

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
        private void GenerateConnections()
        {
            for (int i = 0; i < _layers.Count; i++)
            {
                if (i == 0) continue;

                var curLayer = _layers[i];
                var prevLayer = _layers[i - 1];

                List<Connection> layerConnections = new List<Connection>();

                foreach (var curLayerNeuron in curLayer.Neurons)
                {
                    foreach (var prevLayerNeuron in prevLayer.Neurons)
                    {
                        layerConnections.Add(new Connection(curLayerNeuron, prevLayerNeuron));
                    }
                }

                foreach (var prevLayerNeuron in prevLayer.Neurons)
                {
                    prevLayerNeuron.SetUpForwardConnections(layerConnections.Where(q =>
                        q.BackwardNeuron.NeuronIndex == prevLayerNeuron.NeuronIndex));
                }


                foreach (var curLayerNeuron in curLayer.Neurons)
                {
                    curLayerNeuron.SetUpBackwardConnections(layerConnections.Where(q =>
                        q.ForwardNeuron.NeuronIndex == curLayerNeuron.NeuronIndex));
                }
            }
        }
        private InputLayer GetInputLayer() => _layers.OfType<InputLayer>().Single();
        private OutputLayer GetOutputLayer() => _layers.OfType<OutputLayer>().Single();
        #endregion
    }
}
