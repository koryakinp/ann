using Ann.Activators;
using Ann.Layers;
using Ann.Persistence.LayerConfig;
using Gdo;
using System;
using System.Linq;

namespace Ann
{
    public partial class Network
    {
        public void AddInputLayer(int size, int channels)
        {
            _layers.Add(new InputLayer(new MessageShape(size, channels)));
        }

        public void AddInputLayer(int size)
        {
            _layers.Add(new InputLayer(new MessageShape(size)));
        }

        

        public void AddSoftMaxLayer(Optimizer optimizer)
        {
            var numberOfInputs = _layers
                .Last()
                .OutputMessageShape
                .GetLength();

            var config = new SoftmaxLayerConfiguration(new MessageShape(numberOfInputs));

            var layer = new SoftMaxLayer(config);

            _layers.Add(layer);
        }

        public void AddConvolutionLayer(Optimizer optimizer, int numberOfKernels, int kernelSize)
        {
            var prevLayerOutputShape = _layers
                .Last()
                .OutputMessageShape;

            var layer = new ConvolutionLayer(numberOfKernels, kernelSize, prevLayerOutputShape, optimizer);
            _layers.Add(layer);
        }

        public void AddActivationLayer(ActivatorType activatorType)
        {
            var prevLayerOutputShape = _layers
                .Last()
                .OutputMessageShape;

            var layer = new ActivationLayer(
                prevLayerOutputShape, 
                activatorType);
            _layers.Add(layer);
        }

        public void AddDenseLayer(int numberOfNeurons, Optimizer optimizer)
        {
            var prevLayerOutputShape = _layers
                .Last()
                .OutputMessageShape;

            var config = new DenseLayerConfiguration(prevLayerOutputShape, optimizer, numberOfNeurons);
            var layer = new DenseLayer(config);

            _layers.Add(layer);
        }

        public void AddPoolingLayer(int kernelSize)
        {
            var prevLayerOutputShape = _layers
                .Last()
                .OutputMessageShape;

            var layer = new PoolingLayer(kernelSize, prevLayerOutputShape);
            _layers.Add(layer);
        }

        public void AddFlattenLayer()
        {
            var prevLayerOutputShape = _layers
                .Last()
                .OutputMessageShape;

            var layer = new FlattenLayer(prevLayerOutputShape);
            _layers.Add(layer);
        }
    }
}
