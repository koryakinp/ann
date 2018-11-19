using System;

namespace Ann.Core.Persistence.LayerConfig
{
    [Serializable]
    class ConvolutionLayerConfigurtion : LayerConfiguration
    {
        public readonly int NumberOfKernels;
        public readonly int KernelSize;
        public readonly double[][,,] Weights;
        public readonly double[] Biases;

        public ConvolutionLayerConfigurtion(int numberOfKernels,
            int kernelSize,
            double[][,,] weights,
            double[] biases,
            MessageShape inputMessageShape) : base(inputMessageShape)
        {
            Weights = weights;
            Biases = biases;
            NumberOfKernels = numberOfKernels;
            KernelSize = kernelSize;
        }
    }
}
