using System;

namespace Ann.Core.Layers
{
    public abstract class Layer
    {
        protected readonly MessageShape InputMessageShape;

        public Layer(MessageShape inputMessageShape)
        {
            InputMessageShape = inputMessageShape;
        }

        public abstract MessageShape GetOutputMessageShape();
        public abstract Array PassForward(Array input);
        public abstract Array PassBackward(Array input);
    }
}
