using System;
using Ann.Persistence;
using Ann.Persistence.LayerConfig;

namespace Ann.Layers.Input
{
    class InputForwardLayer : BaseLayer, IForwardLayer
    {
        public InputForwardLayer(
            MessageShape inputMessageShape) 
            : base(inputMessageShape, inputMessageShape) {}

        public LayerConfiguration GetConfiguration()
        {
            return new InputLayerConfiguration(GetInputMessageShape());
        }

        public Array PassForward(Array input)
        {
            return input;
        }
    }
}
