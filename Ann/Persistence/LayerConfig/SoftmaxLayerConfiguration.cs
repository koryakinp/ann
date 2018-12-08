using System;

namespace Ann.Persistence.LayerConfig
{
    [Serializable]
    internal class SoftmaxLayerConfiguration : LayerConfiguration
    {
        public SoftmaxLayerConfiguration(MessageShape inputMessageShape)
            : base(inputMessageShape) { }
    }
}
