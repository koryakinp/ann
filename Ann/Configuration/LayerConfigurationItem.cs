using Ann.Activators;

namespace Ann.Configuration
{
    internal class LayerConfigurationItem
    {
        public readonly LayerType LayerType;
        public readonly int NumberOfNeurons;
        public readonly IActivator Activator;
        public readonly IWeightInitializer WeightInitializer;

        public LayerConfigurationItem(LayerType layerType, int numberOfNeurons, IActivator activator, IWeightInitializer weightInitializer)
        {
            WeightInitializer = weightInitializer;
            Activator = activator;
            LayerType = layerType;
            NumberOfNeurons = numberOfNeurons;
        }

        public LayerConfigurationItem(LayerType layerType, int numberOfNeurons)
        {
            LayerType = layerType;
            NumberOfNeurons = numberOfNeurons;
        }
    }
}
