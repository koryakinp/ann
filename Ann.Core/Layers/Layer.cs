using System;

namespace Ann.Core.Layers
{
    public abstract class Layer
    {
        public readonly MessageShape InputMessageShape;
        public readonly MessageShape OutputMessageShape;

        public Layer(MessageShape inputMessageShape, MessageShape outputMessageShape)
        {
            OutputMessageShape = outputMessageShape;
            InputMessageShape = inputMessageShape;
        }

        public abstract void ValidateForwardInput(Array input);
        public abstract void ValidateBackwardInput(Array input);
        public abstract Array PassForward(Array input);
        public abstract Array PassBackward(Array input);
    }
}
