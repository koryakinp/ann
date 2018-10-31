using System;

namespace Ann.Core.Layers
{
    public class InputLayer : Layer
    {
        public InputLayer(MessageShape inputMessageShape) 
            : base(inputMessageShape, inputMessageShape) {}

        public override Array PassBackward(Array input)
        {
            return input;
        }

        public override Array PassForward(Array input)
        {
            return input;
        }

        public override void ValidateForwardInput(Array input) {}

        public override void ValidateBackwardInput(Array input) {}
    }
}
