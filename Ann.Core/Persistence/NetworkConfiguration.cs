using System;
using System.Collections.Generic;

namespace Ann.Core.Persistence
{
    [Serializable]
    public class NetworkConfiguration
    {
        public readonly List<LayerConfiguration> Layers;

        public NetworkConfiguration()
        {
            Layers = new List<LayerConfiguration>();
        }
    }
}
