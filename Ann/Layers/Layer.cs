using Ann.Persistence;
using System;

namespace Ann.Layers
{
    internal abstract class Layer
    {
        public readonly MessageShape InputMessageShape;
        public readonly MessageShape OutputMessageShape;

        public Layer(MessageShape inputMessageShape, MessageShape outputMessageShape)
        {
            OutputMessageShape = outputMessageShape;
            InputMessageShape = inputMessageShape;
        }

        public virtual void ValidateForwardInput(Array input)
        {
            ValidateInput(input);
        }

        public virtual void ValidateBackwardInput(Array input)
        {
            ValidateInput(input);
        }

        private void ValidateInput(Array input)
        {
            if(input.Length == 0)
            {
                throw new Exception(Consts.CommonLayerMessages.MessageDimenionsInvalid);
            }
        }

        public abstract LayerConfiguration GetLayerConfiguration();

        public abstract Array PassForward(Array input);
        public abstract Array PassBackward(Array input);
    }
}
