using System;

namespace Ann.Persistence.LayerConfig
{
    [Serializable]
    class InputLayerConfiguration : LayerConfiguration
    {
        public InputLayerConfiguration(MessageShape inputMessageShape) 
            : base(inputMessageShape)
        {}
    }
}
