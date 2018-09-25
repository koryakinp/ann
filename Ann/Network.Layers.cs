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
            _layers.Add(new InputLayer(new MessageShape(size, channels)));
        }

        public void AddHiddenLayer(int numberOfNeurons, ActivatorType activator, Optimizer optimizer)
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
                ActivatorFactory.Produce(activator),
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
    }
}
