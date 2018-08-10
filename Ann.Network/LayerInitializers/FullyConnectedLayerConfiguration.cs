using Ann.Activators;
using Ann.Core;
using Ann.Perceptron;
using Gdo;
using System;
using System.Linq;
using Activator = Ann.Activators.Activator;

namespace Ann.Network.LayerInitializers
{
    public class FullyConnectedLayerConfiguration : LayerConfiguration
    {
        private readonly Activator _activator;
        private readonly Optimizer _optimizerCreator;
        private readonly int _numberOfNeurons;

        public FullyConnectedLayerConfiguration(int numberOfNeurons, ActivatorType activator, Optimizer optimizer)
        {
            _activator = ActivatorFactory.Produce(activator);
            _optimizerCreator = optimizer;
            _numberOfNeurons = numberOfNeurons;
        }

        public override ILayer CreateLayer(Network network)
        {
            if(!network._layers.Any())
            {
                throw new Exception(Consts.MissingInputLayer);
            }

            return new FullyConnectedLayer(
                _numberOfNeurons, 
                network._layers.Last().GetNumberOfOutputs(), 
                _activator, 
                _optimizerCreator);
        }
    }
}
