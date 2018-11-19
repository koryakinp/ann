using Ann.Core;
using Ann.Core.Layers;
using Ann.Core.LossFunctions;
using Ann.Core.Persistence;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Ann
{
    public partial class Network
    {
        internal readonly List<Layer> _layers;

        private readonly LossFunction _lossFunction;
        internal readonly int _numberOfClasses;

        private readonly JsonSerializerSettings jsonSerializerSettings = new JsonSerializerSettings
        {
            TypeNameHandling = TypeNameHandling.Auto,
            Formatting = Formatting.Indented
        };

        public Network(string path)
        {
            var json = File.ReadAllText(path);
            var nc = JsonConvert.DeserializeObject<NetworkConfiguration>(json, jsonSerializerSettings);
            _layers = new List<Layer>();
            foreach (var layerConfig in nc.Layers)
            {
                var layer = LayerFactory.Produce(layerConfig);

                _layers.Add(layer);

            }
        }

        public Network(LossFunctionType lossFunctionType, int numberOfClasses)
        {
            _numberOfClasses = numberOfClasses;
            _lossFunction = LossFunctionFactory.Produce(lossFunctionType);
            _layers = new List<Layer>();
        }


        public void TrainModel(Array input, bool[] labels)
        {
            double[] output = PassForward(input).Cast<double>().ToArray();
            double[] error = _lossFunction.ComputeDeriviative(labels, output);

            PassBackward(error);
            Learn();
        }

        public double[] UseModel(Array input)
        {
            return PassForward(input).Cast<double>().ToArray();
        }

        private Array PassForward(Array input)
        {
            for (int i = 0; i < _layers.Count; i++)
            {
                _layers[i].ValidateForwardInput(input);
                input = _layers[i].PassForward(input);
            }

            return input;
        }

        private void PassBackward(Array error)
        {
            for (int i = _layers.Count - 1; i >= 0; i--)
            {
                _layers[i].ValidateBackwardInput(error);
                error = _layers[i].PassBackward(error);
            }
        }

        private void Learn()
        {
            foreach (var item in _layers.OfType<ILearnable>())
            {
                item.UpdateWeights();
                item.UpdateBiases();
            }
        }

        public void RandomizeWeights(double stddev)
        {
            foreach (var layer in _layers.OfType<ILearnable>())
            {
                layer.RandomizeWeights(stddev);
            }
        }

        internal void SetWeights(int layerIndex, Array weights)
        {
            _layers.OfType<ILearnable>().ToArray()[layerIndex].SetWeights(weights);
        }

        public void SaveModel(string path)
        {
            var networkConfig = new NetworkConfiguration();
            foreach (var layer in _layers)
            {
                networkConfig.Layers.Add(layer.GetLayerConfiguration());
            }

            var json = JsonConvert.SerializeObject(networkConfig, jsonSerializerSettings);
            File.WriteAllText(path, json);
        }
    }
}
