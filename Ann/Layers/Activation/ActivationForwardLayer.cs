using Ann.Activators;
using Ann.Persistence;
using Ann.Persistence.LayerConfig;
using Ann.Utils;
using System;
using Activator = Ann.Activators.Activator;

namespace Ann.Layers.Activation
{
    class ActivationForwardLayer : BaseLayer, IForwardLayer
    {
        protected readonly Activator Activator;

        public ActivationForwardLayer(
            ActivatorType type,
            MessageShape inputMessageShape) 
            : base(inputMessageShape, inputMessageShape)
        {
            Activator = ActivatorFactory.Produce(type);
        }

        public LayerConfiguration GetConfiguration()
        {
            return new ActivationLayerConfiguration(
                GetInputMessageShape(),
                Activator.GetActivatorType());
        }

        public Array PassForward(Array input)
        {
            input.UpdateForEach<double>(q => Activator.CalculateValue(q));
            return input;
        }
    }
}
