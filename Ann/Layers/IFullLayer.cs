using System;

namespace Ann.Layers
{
    internal interface IFullLayer : IForwardLayer
    {
        Array PassBackward(Array error);
        MessageShape GetOutputMessageShape();
        MessageShape GetInputMessageShape();
    }
}
