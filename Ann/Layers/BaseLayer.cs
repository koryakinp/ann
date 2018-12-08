namespace Ann.Layers
{
    internal class BaseLayer
    {
        private readonly MessageShape _inputMessageShape;
        private readonly MessageShape _outputMessageShape;

        public BaseLayer(
            MessageShape inputMessageShape, 
            MessageShape outputMessageShape)
        {
            _outputMessageShape = outputMessageShape;
            _inputMessageShape = inputMessageShape;
        }

        public MessageShape GetOutputMessageShape()
        {
            return _outputMessageShape;
        }

        public MessageShape GetInputMessageShape()
        {
            return _inputMessageShape;
        }
    }
}
