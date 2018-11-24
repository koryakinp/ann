using Gdo;
using System;

namespace Ann.Persistence.LayerConfig
{
    [Serializable]
    class ConvolutionLayerConfigurtion : LayerConfiguration
    {
        public readonly int NumberOfKernels;
        public readonly int KernelSize;
        public readonly double[][,,] Weights;
        public readonly double[] Biases;
        public readonly Optimizer Optimizer;

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

        public ConvolutionLayerConfigurtion(int numberOfKernels,
            int kernelSize,
            Optimizer optimizer,
            MessageShape inputMessageShape) : base(inputMessageShape)
        {
            Optimizer = optimizer;
            NumberOfKernels = numberOfKernels;
            KernelSize = kernelSize;
        }
    }
}
