using System;
using System.Collections.Generic;
using System.Text;

namespace Ann.Layers
{
    interface IForwardLayer
    {
        Array PassForward(Array input);
    }
}
