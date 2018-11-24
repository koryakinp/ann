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

        [JsonIgnore]
        public readonly Optimizer Optimizer;

        [JsonConstructor]
        public DenseLayerConfiguration(MessageShape inputMessageShape,
           
            int numberOfNeurons,
            double[,] weights,
            double[] biases) 
            : base(inputMessageShape)
        {
            Optimizer = new PlaceholderOptimizer();
            Weights = weights;
            Biases = biases;
            NumberOfNeurons = numberOfNeurons;
        }

        public DenseLayerConfiguration(
            MessageShape inputMessageShape, 
            Optimizer optimizer,
            int numberOfNeurons)
            : base(inputMessageShape)
        {
            Optimizer = optimizer;
            NumberOfNeurons = numberOfNeurons;
        }
    }
}
