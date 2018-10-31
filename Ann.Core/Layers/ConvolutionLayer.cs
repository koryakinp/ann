using Ann.Core.WeightInitializers;
using Ann.Utils;
using Gdo;
using System;
using System.Linq;

namespace Ann.Core.Layers
{
    public class ConvolutionLayer : KernelLayer, ILearnable
    {
        internal readonly Kernel[] _kernels;
        private readonly int _kernelSize;
        private readonly int _numberOfKernels;
        private readonly double[,,] _cache;
        private readonly double[,,] _gradients;

        public ConvolutionLayer(
            int numberOfKernels, 
            int kernelSize, 
            MessageShape inputMessageShape,
            Optimizer optimizer) : base(
                inputMessageShape, 
                BuildOutputMessageShape(inputMessageShape, kernelSize, numberOfKernels))
        {
            _kernels = new Kernel[numberOfKernels];
            _kernels.UpdateForEach<Kernel>(q => new Kernel(kernelSize, inputMessageShape.Depth, optimizer));
            _cache = new double[InputMessageShape.Depth, InputMessageShape.Size, InputMessageShape.Size];
            _gradients = new double[_numberOfKernels, _kernelSize, _kernelSize];
            _kernelSize = kernelSize;
            _numberOfKernels = numberOfKernels;
        }

        public override Array PassForward(Array input)
        {
            _cache.UpdateForEach<double>((q, idx) => (double)input.GetValue(idx));
            var X = input as double[,,];
            var W = _kernels.Select(q => q.GetValues()).ToArray();
            return MatrixHelper.Convolution(W, X);
        }

        public override Array PassBackward(Array input)
        {
            var gradients = input as double[,,];
            ComputeFilterGradients(gradients);
            return ComputeInputGradients(gradients);
        }

        public void RandomizeWeights(IWeightInitializer weightInitializer)
        {
            _kernels.ForEach(q => q.RandomizeWeights(weightInitializer));
        }

        public void SetWeights(Array weights)
        {
            if(weights.Length != _numberOfKernels)
            {
                throw new Exception(Consts.CommonLayerMessages.CanNotSetWeights);
            }

            _kernels.ForEach((q,kernel) =>
            {
                if(!(weights.GetValue(kernel) is double[,,]))
                {
                    throw new Exception(Consts.CommonLayerMessages.CanNotSetWeights);
                }

                var temp = (double[,,])weights.GetValue(kernel);
                if(temp.Rank != 3 
                    || temp.GetLength(0) != InputMessageShape.Depth
                    || temp.GetLength(1) != _kernelSize
                    || temp.GetLength(2) != _kernelSize)
                {
                    throw new Exception(Consts.CommonLayerMessages.CanNotSetWeights);
                }

                q.Weights.ForEach((w, i, j, k) => w.SetValue(temp[i, j, k]));
            });
        }

        public void UpdateBiases()
        {
            foreach (var kernel in _kernels)
            {
                kernel.UpdateWeights();
            }
        }

        public void UpdateWeights()
        {
            foreach (var kernel in _kernels)
            {
                kernel.UpdateBias();
            }
        }

        private void ComputeFilterGradients(double[,,] gradients)
        {
            for (int i = 0; i < gradients.GetLength(0); i++)
            {
                var gradient = new double[_cache.GetLength(0), _kernelSize, _kernelSize];

                for (int j = 0; j < gradient.GetLength(0); j++)
                {
                    var X = _cache.GetChannel(i);
                    var dEdO = gradients.GetChannel(i);
                    gradient.SetChannel(MatrixHelper.Convolution(X, dEdO), i);
                }

                _kernels[i].SetGradient(gradient);
            }
        }

        private double[,,] ComputeInputGradients(double[,,] input)
        {
            var transposed = _kernels
                .Select(q => q.GetValues())
                .ToArray()
                .Transpose()
                .Select(q => q.Rotate())
                .ToArray();

            var padded = input.Pad(_kernelSize - 1);

            return MatrixHelper.Convolution(transposed, padded);
        }

        public static MessageShape BuildOutputMessageShape(
            MessageShape inputMessageShape, 
            int kernelSize,
            int numberOfKernels)
        {
            int size = inputMessageShape.Size - kernelSize + 1;
            return new MessageShape(size, numberOfKernels);
        }
    }
}
