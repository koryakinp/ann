using Ann.Activators;
using System;

namespace Ann.Persistence.LayerConfig
{
    [Serializable]
    internal class ActivationLayerConfiguration : LayerConfiguration
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
