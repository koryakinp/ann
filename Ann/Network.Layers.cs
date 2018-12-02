using Ann.Activators;
using Ann.Layers;
using Ann.Persistence.LayerConfig;
using Gdo;
using System.Linq;

namespace Ann
{
    public partial class Network
    {
        public void AddInputLayer(int size, int channels)
        {
            var config = new InputLayerConfiguration(new MessageShape(size));
            _layers.Add(new InputLayer(config));
        }

        public void AddInputLayer(int size)
        {
            var config = new InputLayerConfiguration(new MessageShape(size));
            _layers.Add(new InputLayer(config));
        }

        public void AddSoftMaxLayer()
        {
            var shape = _layers
                .Last()
                .OutputMessageShape;

            var config = new SoftmaxLayerConfiguration(shape);

            var layer = new SoftMaxLayer(config);

            _layers.Add(layer);
        }

        public void AddConvolutionLayer(Optimizer optimizer, int numberOfKernels, int kernelSize)
        {
            var prevLayerOutputShape = _layers
                .Last()
                .OutputMessageShape;

            var config = new ConvolutionLayerConfigurtion(numberOfKernels, kernelSize, optimizer, prevLayerOutputShape);

            var layer = new ConvolutionLayer(config);
            _layers.Add(layer);
        }

        public void AddActivationLayer(ActivatorType activatorType)
        {
            var prevLayerOutputShape = _layers
                .Last()
                .OutputMessageShape;

            var config = new ActivationLayerConfiguration(prevLayerOutputShape, activatorType);
            var layer = new ActivationLayer(config);
            _layers.Add(layer);
        }

        public void AddDenseLayer(int numberOfNeurons, bool enableBiases, Optimizer optimizer)
        {
            var prevLayerOutputShape = _layers
                .Last()
                .OutputMessageShape;

            var config = new DenseLayerConfiguration(prevLayerOutputShape, optimizer, enableBiases, numberOfNeurons);
            var layer = new DenseLayer(config);

            _layers.Add(layer);
        }

        public void AddPoolingLayer(int kernelSize)
        {
            var prevLayerOutputShape = _layers
                .Last()
                .OutputMessageShape;

            var config = new PoolingLayerConfiguration(kernelSize, prevLayerOutputShape);

            var layer = new PoolingLayer(config);
            _layers.Add(layer);
        }

        public void AddFlattenLayer()
        {
            var prevLayerOutputShape = _layers
                .Last()
                .OutputMessageShape;

            var config = new FlattenLayerConfiguration(prevLayerOutputShape);

            var layer = new FlattenLayer(config);
            _layers.Add(layer);
        }
    }
}
