using Ann.Activators;
using Ann.Core.WeightInitializers;
using Ann.Utils;
using Gdo;
using System;
using Activator = Ann.Activators.Activator;

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
            var output = new double[OutputMessageShape.Depth, OutputMessageShape.Size, OutputMessageShape.Size];
            _cache.UpdateForEach<double>((q,idx) => (double)input.GetValue(idx));

            for (int i = 0; i < _kernels.Length; i++)
            {
                double[,] temp = MatrixHelper.Convolution(input as double[,,], _kernels[i].Weights.Values());
                temp.ForEach((q, j, k) => output[i, j, k] = q);
            }

            return output;
        }

        public override Array PassBackward(Array input)
        {
            //var transposed = MatrixHelper.Transpose(_kernels.Values());
            ////var deltas = MatrixHelper.Convolution(_cache, transposed);
            ////_gradients.UpdateForEach<double>((q,idx) => (double)deltas.GetValue(idx));

            //var flipped = MatrixHelper.Flip(transposed);
            //var padded = MatrixHelper.Pad(input as double[,,], _kernelSize - 1);
            //var output = MatrixHelper.Convolution(padded, flipped);
            //return output;

            throw new NotImplementedException();
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
            throw new System.NotImplementedException();
        }

        public void UpdateWeights()
        {
            throw new System.NotImplementedException();
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
