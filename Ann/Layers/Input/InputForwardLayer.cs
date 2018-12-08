using System;

namespace Ann.Layers.Input
{
    class InputForwardLayer : BaseLayer, IForwardLayer
    {
        public InputForwardLayer(
            MessageShape inputMessageShape) 
            : base(inputMessageShape, inputMessageShape) {}

        public Array PassForward(Array input)
        {
            return input;
        }
    }
}
