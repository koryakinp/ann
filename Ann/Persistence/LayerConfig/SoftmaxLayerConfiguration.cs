using System;

namespace Ann.Persistence.LayerConfig
{
    [Serializable]
    class SoftmaxLayerConfiguration : LayerConfiguration
    {
        public SoftmaxLayerConfiguration(MessageShape inputMessageShape)
            : base(inputMessageShape) { }
    }
}
