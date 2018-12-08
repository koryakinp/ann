using Ann.Persistence;
using System;

namespace Ann.Layers
{
    internal interface IForwardLayer
    {
        Array PassForward(Array input);
        LayerConfiguration GetConfiguration();
    }
}
