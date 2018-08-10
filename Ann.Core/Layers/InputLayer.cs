namespace Ann.Core.Layers
{
    public class InputLayer : Layer
    {
        public InputLayer(MessageShape inputMessageShape) 
            : base(inputMessageShape) {}

        public override Message PassBackward(Message input)
        {
            return input;
        }

        public override Message PassForward(Message input)
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
