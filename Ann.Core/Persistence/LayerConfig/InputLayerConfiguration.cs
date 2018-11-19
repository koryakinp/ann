using System;

namespace Ann.Core.Persistence.LayerConfig
{
    [Serializable]
    class InputLayerConfiguration : LayerConfiguration
    {
        public InputLayerConfiguration(MessageShape inputMessageShape) 
            : base(inputMessageShape) {}
    }
}
