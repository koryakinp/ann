using System;

namespace Ann.Layers
{
    interface IFullLayer : IForwardLayer
    {
        Array PassBackward(Array error);
        MessageShape GetOutputMessageShape();
        MessageShape GetInputMessageShape();
    }
}
