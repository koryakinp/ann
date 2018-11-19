using Ann.Activators;
using System;

namespace Ann.Core.Persistence.LayerConfig
{
    [Serializable]
    class ActivationLayerConfiguration : LayerConfiguration
    {
        public readonly ActivatorType ActivatorType;

        public ActivationLayerConfiguration(
            MessageShape inputMessageShape, 
            ActivatorType activatorType) 
            : base(inputMessageShape)
        {

            ActivatorType = activatorType;
        }
    }
}
