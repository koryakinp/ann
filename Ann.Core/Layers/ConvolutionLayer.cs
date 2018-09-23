using Ann.Activators;
using Ann.Core.WeightInitializers;
using Ann.Utils;
using Gdo;
using System;
using Activator = Ann.Activators.Activator;

namespace Ann.Core.Layers
{
    public class ConvolutionLayer : Layer, ILearnable
    {
        private readonly Optimizer[,,,] _kernels;
        private readonly int _kernelSize;
        private readonly int _numberOfKernels;
        private readonly Activator _activator;
        private readonly Optimizer[] _biases;
        private readonly double[,,] _cache;
        private readonly double[,,] _gradients;

        public ConvolutionLayer(
            int numberOfKernels, 
            int kernelSize, 
            MessageShape inputMessageShape,
            Optimizer optimizer,
            ActivatorType activator) : base(inputMessageShape)
        {

            _biases = new Optimizer[numberOfKernels];
            _biases.ForEach((q,i) => _biases[i] = optimizer.Clone() as Optimizer);
            _activator = ActivatorFactory.Produce(activator);
            _kernels = new Optimizer[numberOfKernels, InputMessageShape.Depth, kernelSize, kernelSize];
            _cache = new double[InputMessageShape.Depth, InputMessageShape.Size, InputMessageShape.Size];
            _gradients = new double[_numberOfKernels, _kernelSize, _kernelSize];
            _kernels.ForEach((i, j, k, p) => _kernels[i, j, k, p] = optimizer.Clone() as Optimizer);
            _kernelSize = kernelSize;
            _numberOfKernels = numberOfKernels;
        }

        public override Array PassForward(Array input)
        {
            //_cache.UpdateForEach<double>((q,idx) => (double)input.GetValue(idx));
            //var res = MatrixHelper.Convolution(input as double[,,], _kernels.Values());
            //res.UpdateForEach<double>((q,idx) => _activator.CalculateValue(q + _biases[idx[0]].Value));
            //return res;
            throw new NotImplementedException();
        }

        public override Array PassBackward(Array input)
        {
            //input.UpdateForEach<double>((q) => _activator.CalculateDeriviative(q));
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
            double magnitude = 1 / (_kernelSize * _kernelSize * InputMessageShape.Depth);
            _kernels.ForEach((q,k,d,i,j) => q.SetValue(weightInitializer.GenerateRandom(magnitude)));
        }

        public void UpdateBiases()
        {
            throw new System.NotImplementedException();
        }

        public void UpdateWeights()
        {
            throw new System.NotImplementedException();
        }

        public override MessageShape GetOutputMessageShape()
        {
            int size = InputMessageShape.Size - _kernelSize + 1;
            return new MessageShape(size, _numberOfKernels);
        }
    }
}
