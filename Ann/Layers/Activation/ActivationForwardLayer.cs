using Ann.Activators;
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

        public Array PassForward(Array input)
        {
            input.UpdateForEach<double>(q => Activator.CalculateValue(q));
            return input;
        }
    }
}
