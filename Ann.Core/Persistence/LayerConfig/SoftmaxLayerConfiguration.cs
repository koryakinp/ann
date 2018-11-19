using System;

namespace Ann.Core.Persistence.LayerConfig
{
    [Serializable]
    class SoftmaxLayerConfiguration : LayerConfiguration
    {
        public readonly int NumberOfNeurons;
        public readonly double[][] Weights;

        public SoftmaxLayerConfiguration(
            int numberOfNeurons,
            double[][] weights,
            MessageShape inputMessageShape) 
            : base(inputMessageShape)
        {
            Weights = weights;
            NumberOfNeurons = numberOfNeurons;
        }
    }
}
