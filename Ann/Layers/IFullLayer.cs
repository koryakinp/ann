using System;
using System.Collections.Generic;
using System.Text;

namespace Ann.Layers
{
    interface IFullLayer : IForwardLayer
    {
        Array PassBackward(Array error);
    }
}
