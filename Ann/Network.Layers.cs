﻿using Ann.Activators;
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

        public void AddHiddenLayer(int numberOfNeurons, ActivatorType activatorType, Optimizer optimizer)
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

        public void AddSoftMaxLayer(Optimizer optimizer)
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
            var shape = _layers
                .Last()
                .OutputMessageShape;

            var config = new ActivationLayerConfiguration(shape, activatorType);
            var layer = new ActivationLayer(config);
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
            var shape = _layers
                .Last()
                .OutputMessageShape;

            var config = new FlattenLayerConfiguration(shape);

            var layer = new FlattenLayer(config);
            _layers.Add(layer);
        }
    }
}
