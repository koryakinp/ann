using Ann.Activators;
using Ann.Core;
using Ann.Core.Layers;
using Gdo;
using System;
using System.Linq;

namespace Ann
{
    public partial class Network
    {
        public void AddInputLayer(int size, int channels)
        {
            _layers.Add(new InputLayer(new MessageShape(size, size, channels)));
        }

        public void AddHiddenLayer(int numberOfNeurons, ActivatorType activator, Optimizer optimizer)
        {
            if (!_layers.Any())
            {
                throw new Exception(Consts.MissingInputLayer);
            }

            var numberOfInputs = _layers
                .Last()
                .GetOutputMessageShape()
                .GetLength();

            var layer = new HiddenLayer(
                numberOfNeurons,
                ActivatorFactory.Produce(activator),
                optimizer,
                new MessageShape(1, numberOfInputs, 1));

            _layers.Add(layer);
        }

        public void AddSoftMaxLayer(Optimizer optimizer)
        {

            var numberOfInputs = _layers
                .Last()
                .GetOutputMessageShape()
                .GetLength();

            var layer = new SoftMaxLayer(
                _numberOfClasses,
                new MessageShape(1, numberOfInputs, 1),
                optimizer);

            _layers.Add(layer);
        }
    }
}
