using System;

namespace Ann.Persistence.LayerConfig
{
    [Serializable]
    internal class InputLayerConfiguration : LayerConfiguration
    {
        public InputLayerConfiguration(MessageShape inputMessageShape) 
            : base(inputMessageShape)
        {}
    }
}
