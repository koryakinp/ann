using System;

namespace Ann.Persistence.LayerConfig
{
    [Serializable]
    class FlattenLayerConfiguration : LayerConfiguration
    {
        public FlattenLayerConfiguration(MessageShape inputMessageShape) 
            : base(inputMessageShape) {}
    }
}
