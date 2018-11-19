using System;

namespace Ann.Layers
{
    internal abstract class KernelLayer : Layer
    {
        public KernelLayer(MessageShape inputMessageShape, MessageShape outputMessageShape) 
            : base(inputMessageShape, outputMessageShape) {}

        public override void ValidateBackwardInput(Array input)
        {
            base.ValidateForwardInput(input);
            ValidateInput(input, OutputMessageShape);
        }

        public override void ValidateForwardInput(Array input)
        {
            base.ValidateBackwardInput(input);
            ValidateInput(input, InputMessageShape);
        }

        private void ValidateInput(Array input, MessageShape shape)
        {
            if (shape.Size != input.GetLength(1)
                || shape.Size != input.GetLength(2)
                || shape.Depth != input.GetLength(0))
            {
                throw new Exception(Consts.CommonLayerMessages.MessageDimenionsInvalid);
            }
        }
    }
}
