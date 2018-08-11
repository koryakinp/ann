using System;

namespace Ann.Core.Layers
{
    public class InputLayer : Layer
    {
        public InputLayer(MessageShape inputMessageShape) 
            : base(inputMessageShape) {}

        public override Array PassBackward(Array input)
        {
            return input;
        }

        public override Array PassForward(Array input)
        {
            return input;
        }

        public override MessageShape GetOutputMessageShape()
        {
            return new MessageShape(
                InputMessageShape.Width, 
                InputMessageShape.Height, 
                InputMessageShape.Depth);
        }
    }
}
