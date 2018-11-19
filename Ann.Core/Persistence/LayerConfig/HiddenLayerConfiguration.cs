using Ann.Activators;
using System;

namespace Ann.Core.Persistence.LayerConfig
{
    [Serializable]
    class HiddenLayerConfiguration : LayerConfiguration
    {
        public readonly ActivatorType ActivatorType;
        public readonly int NumberOfNeurons;
        public readonly double[][] Weights;
        public readonly double[] Biases;

        public HiddenLayerConfiguration(
            MessageShape inputMessageShape,
            int numberOfNeurons,
            double[][] weights,
            double[] biases,
            ActivatorType activatorType) 
            : base(inputMessageShape)
        {
            Weights = weights;
            Biases = biases;
            ActivatorType = activatorType;
            NumberOfNeurons = numberOfNeurons;
        }
    }
}
