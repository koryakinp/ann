using Ann.Activators;
using System;

namespace Ann.Layers.SoftMax
{
    class SoftMaxForwardLayer : BaseLayer, IForwardLayer
    {
        protected readonly SoftmaxActivator Activator = new SoftmaxActivator();

        public SoftMaxForwardLayer(
            MessageShape inputMessageShape) 
            : base(inputMessageShape, inputMessageShape) {}

        public Array PassForward(Array input)
        {
            return Activator.CalculateValue(input as double[]);
        }
    }
}
