using Ann.Persistence;
using System;

namespace Ann.Layers
{
    interface IForwardLayer
    {
        Array PassForward(Array input);
        LayerConfiguration GetConfiguration();
    }
}
