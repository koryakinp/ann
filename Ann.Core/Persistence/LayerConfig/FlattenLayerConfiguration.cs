using System;

namespace Ann.Core.Persistence.LayerConfig
{
    [Serializable]
    class FlattenLayerConfiguration : LayerConfiguration
    {
        public FlattenLayerConfiguration(MessageShape inputMessageShape) 
            : base(inputMessageShape) {}
    }
}
