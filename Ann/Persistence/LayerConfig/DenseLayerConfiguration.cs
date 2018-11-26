using Ann.Misc;
using Gdo;
using Newtonsoft.Json;
using System;

namespace Ann.Persistence.LayerConfig
{
    [Serializable]
    class DenseLayerConfiguration : LayerConfiguration
    {
        public readonly int NumberOfNeurons;
        public readonly double[,] Weights;
        public readonly double[] Biases;
        public readonly bool EnableBias;

        [JsonIgnore]
        public readonly Optimizer Optimizer;

        [JsonConstructor]
        public DenseLayerConfiguration(MessageShape inputMessageShape,
            bool enableBiase,
            int numberOfNeurons,
            double[,] weights,
            double[] biases) 
            : base(inputMessageShape)
        {
            EnableBias = enableBiase;
            Optimizer = new PlaceholderOptimizer();
            Weights = weights;
            Biases = biases;
            NumberOfNeurons = numberOfNeurons;
        }

        public DenseLayerConfiguration(
            MessageShape inputMessageShape, 
            Optimizer optimizer,
            bool enableBiase,
            int numberOfNeurons)
            : base(inputMessageShape)
        {
            EnableBias = enableBiase;
            Optimizer = optimizer;
            NumberOfNeurons = numberOfNeurons;
        }
    }
}
