using Ann.Activators;
using Ann.Persistence;
using Ann.Persistence.LayerConfig;
using System;

namespace Ann.Layers.SoftMax
{
    internal class SoftMaxForwardLayer : BaseLayer, IForwardLayer
    {
        protected readonly SoftmaxActivator Activator = new SoftmaxActivator();

        public SoftMaxForwardLayer(
            MessageShape inputMessageShape) 
            : base(inputMessageShape, inputMessageShape) {}

        public LayerConfiguration GetConfiguration()
        {
            return new SoftmaxLayerConfiguration(GetInputMessageShape());
        }

        public Array PassForward(Array input)
        {
            return Activator.CalculateValue(input as double[]);
        }
    }
}
