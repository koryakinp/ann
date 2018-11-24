using Gdo;
using System;

namespace Ann.Persistence.LayerConfig
{
    [Serializable]
    class DenseLayerConfiguration : LayerConfiguration
    {
        public readonly int NumberOfNeurons;
        public readonly double[,] Weights;
        public readonly double[] Biases;
        public readonly Optimizer Optimizer;

        public DenseLayerConfiguration(MessageShape inputMessageShape,
           
            int numberOfNeurons,
            double[,] weights,
            double[] biases) 
            : base(inputMessageShape)
        {
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
