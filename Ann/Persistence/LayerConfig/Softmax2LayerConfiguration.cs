using System;

namespace Ann.Persistence.LayerConfig
{
    [Serializable]
    class SoftmaxLayer2Configuration : LayerConfiguration
    {
        public SoftmaxLayer2Configuration(MessageShape inputMessageShape)
            : base(inputMessageShape) { }
    }
}