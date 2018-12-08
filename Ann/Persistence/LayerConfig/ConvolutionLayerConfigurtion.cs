using Gdo;
using Newtonsoft.Json;
using System;

namespace Ann.Persistence.LayerConfig
{
    [Serializable]
    internal class ConvolutionLayerConfigurtion : LayerConfiguration
    {
        public readonly int NumberOfKernels;
        public readonly int KernelSize;
        public readonly double[][,,] Weights;
        public readonly double[] Biases;

        [JsonConstructor]
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
            MessageShape inputMessageShape) : base(inputMessageShape)
        {
            NumberOfKernels = numberOfKernels;
            KernelSize = kernelSize;
        }
    }
}
