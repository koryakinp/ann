using System;

namespace Ann.Core.Persistence.LayerConfig
{
    [Serializable]
    class PoolingLayerConfiguration : LayerConfiguration
    {
        public readonly int KernelSize;

        public PoolingLayerConfiguration(int kernelSize, MessageShape inputMessageShape) 
            : base(inputMessageShape)
        {
            KernelSize = kernelSize;
        }
    }
}
