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
        private double[,,] _cache { get; set; }

        public ConvolutionLayer(
            int numberOfKernels, 
            int kernelSize, 
            MessageShape inputMessageShape,
            Optimizer optimizer,
            ActivatorType activator) : base(inputMessageShape)
        {
            if(InputMessageShape.Width != inputMessageShape.Height)
            {
                throw new Exception(Consts.MessageShapeIsNotValid);
            }

            _biases = new Optimizer[numberOfKernels];
            _biases.ForEach((q,i) => _biases[i] = optimizer.Clone() as Optimizer);

            _kernels = new Optimizer[
                numberOfKernels, 
                InputMessageShape.Depth, 
                kernelSize, 
                kernelSize];

            _kernels.ForEach((i, j, k, p) => _kernels[i, j, k, p] = optimizer.Clone() as Optimizer);
            _kernelSize = kernelSize;
            _numberOfKernels = numberOfKernels;
        }

        public override Array PassForward(Array input)
        {
            var res = MatrixHelper2.Convolution(input, _kernels.Values());
            res.ForEach((q, i, j, k) => res[i, j, k] = _activator.CalculateValue(q + _biases[i].Value));
            _cache = res;
            return res;
        }

        public override Array PassBackward(Array input)
        {
            throw new System.NotImplementedException();
        }

        public void RandomizeWeights(IWeightInitializer weightInitializer)
        {
            double magnitude = 1 / (_kernelSize * _kernelSize * InputMessageShape.Depth);
            weightInitializer.GenerateRandom(magnitude);
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
            int size = InputMessageShape.Width - _kernelSize + 1;
            return new MessageShape(size, size, _numberOfKernels);
        }
    }
}
