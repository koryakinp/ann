namespace Ann.Core.Persistence
{
    public abstract class LayerConfiguration
    {
        public readonly MessageShape MessageShape;

        public LayerConfiguration(MessageShape inputMessageShape)
        {
            MessageShape = inputMessageShape;
        }
    }
}
