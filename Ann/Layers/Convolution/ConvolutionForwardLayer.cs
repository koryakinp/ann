using Ann.Persistence;
using Ann.Persistence.LayerConfig;
using Ann.Utils;
using System;

namespace Ann.Layers.Convolution
{
    class ConvolutionForwardLayer : BaseLayer, IForwardLayer
    {
        protected readonly int KernelSize;
        protected readonly int NumberOfKernels;
        protected readonly double[][,,] Kernels;
        protected readonly double[] Biases;

        public ConvolutionForwardLayer(
            MessageShape inputMessageShape,
            int kernelSize,
            int numberOfKernels) : 
            base(inputMessageShape, BuildOutputMessageShape(inputMessageShape, kernelSize, numberOfKernels))
        {
            KernelSize = kernelSize;
            NumberOfKernels = numberOfKernels;
            Kernels = new double[numberOfKernels][,,];
            Kernels.UpdateForEach<double[,,]>(q => new double[inputMessageShape.Depth, kernelSize, kernelSize]);
            Biases = new double[numberOfKernels];
        }

        public Array PassForward(Array input)
        {
            var X = input as double[,,];
            var res = MatrixHelper.Convolution(Kernels, X);
            res.UpdateForEach<double>((q, idx) => q + Biases[idx[0]]);
            return res;
        }

        public static MessageShape BuildOutputMessageShape(
            MessageShape inputMessageShape,
            int kernelSize,
            int numberOfKernels)
        {
            int size = inputMessageShape.Size - kernelSize + 1;
            return new MessageShape(size, numberOfKernels);
        }

        public void SetBiases(double[] biases)
        {
            biases.ForEach((q, i) => Biases[i] = q);
        }

        public void SetWeights(Array weights)
        {
            if (weights.Length != NumberOfKernels)
            {
                throw new Exception(Consts.CommonLayerMessages.CanNotSetWeights);
            }

            Kernels.ForEach((q, kernel) =>
            {
                if (!(weights.GetValue(kernel) is double[,,]) 
                    || q.Rank != 3
                    || q.GetLength(0) != (weights.GetValue(kernel) as double[,,]).GetLength(0)
                    || q.GetLength(1) != (weights.GetValue(kernel) as double[,,]).GetLength(1)
                    || q.GetLength(2) != (weights.GetValue(kernel) as double[,,]).GetLength(2))
                {
                    throw new Exception(Consts.CommonLayerMessages.CanNotSetWeights);
                }

                (weights.GetValue(kernel) as double[,,])
                    .ForEach((w, i, j, k) => Kernels[kernel][i,j,k] = w);
            });
        }

        public LayerConfiguration GetConfiguration()
        {
            return new ConvolutionLayerConfigurtion(
                NumberOfKernels,
                KernelSize,
                Kernels,
                Biases,
                GetInputMessageShape());
        }
    }
}
