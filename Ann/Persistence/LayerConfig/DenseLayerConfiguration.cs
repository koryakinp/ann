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

        [JsonConstructor]
        public DenseLayerConfiguration(MessageShape inputMessageShape,
            bool enableBiase,
            int numberOfNeurons,
            double[,] weights,
            double[] biases) 
            : base(inputMessageShape)
        {
            EnableBias = enableBiase;
            Weights = weights;
            Biases = biases;
            NumberOfNeurons = numberOfNeurons;
        }

        public DenseLayerConfiguration(
            MessageShape inputMessageShape, 
            bool enableBiase,
            int numberOfNeurons)
            : base(inputMessageShape)
        {
            EnableBias = enableBiase;
            NumberOfNeurons = numberOfNeurons;
        }
    }
}
