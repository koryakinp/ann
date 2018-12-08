using Ann.Layers;
using Ann.Persistence;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;

namespace Ann
{
    public class Model
    {
        private readonly List<IForwardLayer> _layers;
        private readonly List<LayerConfiguration> _layerConfiguration;
        private readonly JsonSerializerSettings jsonSerializerSettings = new JsonSerializerSettings
        {
            TypeNameHandling = TypeNameHandling.Auto,
            Formatting = Formatting.Indented
        };

        internal Model(List<LayerConfiguration> layerConfig)
        {
            _layerConfiguration = layerConfig;
            _layers = new List<IForwardLayer>();
            foreach (var lc in _layerConfiguration)
            {
                var layer = LayerFactory.Produce(lc);
                _layers.Add(layer);
            }
        }

        public Model(string path)
        {
            var json = File.ReadAllText(path);
            _layerConfiguration = JsonConvert.DeserializeObject<List<LayerConfiguration>>(json, jsonSerializerSettings);
            _layers = new List<IForwardLayer>();
            foreach (var lc in _layerConfiguration)
            {
                var layer = LayerFactory.Produce(lc);
                _layers.Add(layer);
            }
        }

        public double[] Predict(Array input)
        {
            foreach (var layer in _layers)
            {
                input = layer.PassForward(input);
            }

            return input as double[];
        }

        public void Save(string path)
        {
            var json = JsonConvert.SerializeObject(_layerConfiguration, jsonSerializerSettings);
            File.WriteAllText(path, json);
        }
    }
}
