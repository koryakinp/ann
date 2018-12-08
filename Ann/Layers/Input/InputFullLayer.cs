using System;

namespace Ann.Layers.Input
{
    class InputFullLayer : InputForwardLayer, IFullLayer
    {
        public InputFullLayer(MessageShape inputMessageShape) 
            : base(inputMessageShape) {}

        public Array PassBackward(Array error)
        {
            return error;
        }
    }
}
