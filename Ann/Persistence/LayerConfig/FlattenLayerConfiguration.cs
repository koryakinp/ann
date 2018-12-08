using System;

namespace Ann.Persistence.LayerConfig
{
    [Serializable]
    internal class FlattenLayerConfiguration : LayerConfiguration
    {
        public FlattenLayerConfiguration(MessageShape inputMessageShape) 
            : base(inputMessageShape) {}
    }
}
