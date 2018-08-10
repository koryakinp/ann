using Ann.Core;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ann.Network
{
    public abstract class LayerConfiguration
    {
        public abstract ILayer CreateLayer(Network network);
    }
}
