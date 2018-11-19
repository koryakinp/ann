using System;
using System.Collections.Generic;

namespace Ann.Persistence
{
    [Serializable]
    internal class NetworkConfiguration
    {
        public readonly List<LayerConfiguration> Layers;

        public NetworkConfiguration()
        {
            Layers = new List<LayerConfiguration>();
        }
    }
}
