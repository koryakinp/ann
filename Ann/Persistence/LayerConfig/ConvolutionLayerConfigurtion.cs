using Ann.Misc;
using Gdo;
using Newtonsoft.Json;
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
        [JsonIgnore]
        public readonly Optimizer Optimizer;

        [JsonConstructor]
        public ConvolutionLayerConfigurtion(int numberOfKernels,
            int kernelSize,
            double[][,,] weights,
            double[] biases,
            MessageShape inputMessageShape) : base(inputMessageShape)
        {
            //Optimizer = new PlaceholderOptimizer();
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
