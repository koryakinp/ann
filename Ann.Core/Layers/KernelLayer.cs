using System;

namespace Ann.Core.Layers
{
    public abstract class KernelLayer : Layer
    {
        public KernelLayer(MessageShape inputMessageShape, MessageShape outputMessageShape) 
            : base(inputMessageShape, outputMessageShape) {}

        public override void ValidateBackwardInput(Array input)
        {
            if (OutputMessageShape.Size != input.GetLength(1)
                || OutputMessageShape.Size != input.GetLength(2)
                || OutputMessageShape.Depth != input.GetLength(0))
            {
                throw new Exception(Consts.CommonLayerMessages.MessageDimenionsInvalid);
            }
        }

        public override void ValidateForwardInput(Array input)
        {
            if (InputMessageShape.Size != input.GetLength(1)
                || InputMessageShape.Size != input.GetLength(2)
                || InputMessageShape.Depth != input.GetLength(0))
            {
                throw new Exception(Consts.CommonLayerMessages.MessageDimenionsInvalid);
            }
        }
    }
}
