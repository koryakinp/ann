using Ann.Activators;
using Ann.Layers.Activation;
using Ann.Layers.Convolution;
using Ann.Layers.Dense;
using Ann.Layers.Flatten;
using Ann.Layers.Input;
using Ann.Layers.Pooling;
using Ann.Layers.SoftMax;
using System.Linq;

namespace Ann
{
    public partial class Network
    {
        public void AddInputLayer(int size, int channels)
        {
            _layers.Add(new InputFullLayer(new MessageShape(size, channels)));
        }

        public void AddInputLayer(int size)
        {
            _layers.Add(new InputFullLayer(new MessageShape(size)));
        }

        public void AddSoftMaxLayer()
        {
            var shape = _layers.Last().GetOutputMessageShape();
            _layers.Add(new SoftMaxFullLayer(shape));
        }

        public void AddConvolutionLayer(int numberOfKernels, int kernelSize)
        {
            var shape = _layers.Last().GetOutputMessageShape();
            _layers.Add(new ConvolutionFullLayer(shape, kernelSize, numberOfKernels, _optimizer));
        }

        public void AddActivationLayer(ActivatorType activatorType)
        {
            var shape = _layers.Last().GetOutputMessageShape();
            _layers.Add(new ActivationFullLayer(activatorType, shape));
        }

        public void AddDenseLayer(int numberOfNeurons, bool enableBiases)
        {
            var shape = _layers.Last().GetOutputMessageShape();
            _layers.Add(new DenseFullLayer(shape, numberOfNeurons, enableBiases, _optimizer));
        }

        public void AddPoolingLayer(int kernelSize)
        {
            var shape = _layers.Last().GetOutputMessageShape();
            _layers.Add(new PoolingFullLayer(shape, kernelSize));
        }

        public void AddFlattenLayer()
        {
            var shape = _layers.Last().GetOutputMessageShape();
            _layers.Add(new FlattenFullLayer(shape));
        }
    }
}
