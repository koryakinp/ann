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
        public void AddInputLayer(int size, int channels, bool test = true)
        {
            if (test)
            {
                _layers.Add(new InputLayer(new MessageShape(size, channels)));
            }
            else
            {
                _testLayers.Add(new InputLayer(new MessageShape(size, channels)));
            }
        }

        public void AddInputLayer(int size, bool test = true)
        {
            if (test)
            {
                _layers.Add(new InputLayer(new MessageShape(size)));
            }
            else
            {
                _testLayers.Add(new InputLayer(new MessageShape(size)));
            }
        }

        public void AddHiddenLayer(int numberOfNeurons, ActivatorType activatorType, Optimizer optimizer, bool test = true)
        {
            if (test)
            {
                if (!_layers.Any())
                {
                    throw new Exception(Consts.MissingInputLayer);
                }

                var numberOfInputs = _layers
                    .Last()
                    .OutputMessageShape
                    .GetLength();

                var layer = new HiddenLayer(
                    numberOfNeurons,
                    activatorType,
                    optimizer,
                    new MessageShape(numberOfInputs));

                _layers.Add(layer);
            }
            else
            {
                if (!_layers.Any())
                {
                    throw new Exception(Consts.MissingInputLayer);
                }

                var numberOfInputs = _testLayers
                    .Last()
                    .OutputMessageShape
                    .GetLength();

                var layer = new HiddenLayer(
                    numberOfNeurons,
                    activatorType,
                    optimizer,
                    new MessageShape(numberOfInputs));

                _testLayers.Add(layer);
            }
        }

        public void AddSoftMaxLayer(Optimizer optimizer, bool test = true)
        {
            if (test)
            {
                var numberOfInputs = _layers
                    .Last()
                    .OutputMessageShape
                    .GetLength();

                var layer = new SoftMaxLayer(
                    _numberOfClasses,
                    new MessageShape(numberOfInputs),
                    optimizer);

                _layers.Add(layer);
            }
            else
            {
                var numberOfInputs = _testLayers
                .Last()
                .OutputMessageShape
                .GetLength();

                var layer = new SoftMaxLayer(
                    _numberOfClasses,
                    new MessageShape(numberOfInputs),
                    optimizer);

                _testLayers.Add(layer);
            }
        }

        public void AddSoftMaxLayer2(bool test = true)
        {
            if (test)
            {
                var shape = _layers
                    .Last()
                    .OutputMessageShape;

                var config = new SoftmaxLayer2Configuration(shape);
                var layer = new SoftMaxLayer2(config);
                _layers.Add(layer);
            }
            else
            {
                var shape = _layers
                    .Last()
                    .OutputMessageShape;

                var config = new SoftmaxLayer2Configuration(shape);
                var layer = new SoftMaxLayer2(config);
                _testLayers.Add(layer);
            }
        }

        public void AddConvolutionLayer(Optimizer optimizer, int numberOfKernels, int kernelSize, bool test = true)
        {
            if (test)
            {
                var prevLayerOutputShape = _layers
                    .Last()
                    .OutputMessageShape;

                var layer = new ConvolutionLayer(numberOfKernels, kernelSize, prevLayerOutputShape, optimizer);
                _layers.Add(layer);
            }
            else
            {
                var prevLayerOutputShape = _testLayers
                    .Last()
                    .OutputMessageShape;

                var layer = new ConvolutionLayer(numberOfKernels, kernelSize, prevLayerOutputShape, optimizer);
                _testLayers.Add(layer);
            }
        }

        public void AddActivationLayer(ActivatorType activatorType, bool test = true)
        {
            if(test)
            {
                var shape = _layers
                    .Last()
                    .OutputMessageShape;

                var config = new ActivationLayerConfiguration(shape, activatorType);
                var layer = new ActivationLayer(config);
                _layers.Add(layer);
            }
            else
            {
                var shape = _testLayers
                    .Last()
                    .OutputMessageShape;

                var config = new ActivationLayerConfiguration(shape, activatorType);
                var layer = new ActivationLayer(config);
                _testLayers.Add(layer);
            }
        }

        public void AddPoolingLayer(int kernelSize, bool test = true)
        {
            if(test)
            {
                var prevLayerOutputShape = _layers
                    .Last()
                    .OutputMessageShape;

                var layer = new PoolingLayer(kernelSize, prevLayerOutputShape);
                _layers.Add(layer);
            }
            else
            {
                var prevLayerOutputShape = _testLayers
                    .Last()
                    .OutputMessageShape;

                var layer = new PoolingLayer(kernelSize, prevLayerOutputShape);
                _testLayers.Add(layer);
            }
        }

        public void AddFlattenLayer(bool test = true)
        {
            if(test)
            {
                var shape = _layers
                    .Last()
                    .OutputMessageShape;

                var config = new FlattenLayerConfiguration(shape);

                var layer = new FlattenLayer(config);
                _layers.Add(layer);
            }
            else
            {
                var shape = _testLayers
                    .Last()
                    .OutputMessageShape;

                var config = new FlattenLayerConfiguration(shape);

                var layer = new FlattenLayer(config);
                _testLayers.Add(layer);
            }
        }

        public void AddDenseLayer(int numberOfNeurons, bool enableBiases, Optimizer optimizer, bool test = true)
        {
            if(test)
            {
                var prevLayerOutputShape = _layers
                    .Last()
                    .OutputMessageShape;

                var config = new DenseLayerConfiguration(prevLayerOutputShape, optimizer, enableBiases, numberOfNeurons);
                var layer = new DenseLayer(config);

                _layers.Add(layer);
            }
            else
            {
                var prevLayerOutputShape = _testLayers
                    .Last()
                    .OutputMessageShape;

                var config = new DenseLayerConfiguration(prevLayerOutputShape, optimizer, enableBiases, numberOfNeurons);
                var layer = new DenseLayer(config);

                _testLayers.Add(layer);
            }
        }
    }
}
