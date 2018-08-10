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

        public override Message PassBackward(Message input)
        {
            var data = input.ToMulti(GetOutputMessageShape().Depth, GetOutputMessageShape().Width);
            data.ForEach((q, i, j, k) => data[i, j, k] = q * _activator.CalculateDeriviative(_cache[i,j,k]));


            throw new System.NotImplementedException();
        }

        public override Message PassForward(Message input)
        {
            if(!input.IsMulti)
            {
                throw new Exception(Consts.ConvolutionalInputIsNotValid);
            }

            var data = input.ToMulti(InputMessageShape.Depth, InputMessageShape.Width);
            var res = MatrixHelper.Convolution(data, _kernels.Values());
            res.ForEach((q, i, j, k) => res[i, j, k] += _biases[i].Value);
            res.ForEach((q, i, j, k) => res[i, j, k] = _activator.CalculateValue(q));
            _cache = res;
            return new Message(res);
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
