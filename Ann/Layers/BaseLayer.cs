using System;
using System.Collections.Generic;
using System.Text;

namespace Ann.Layers
{
    class BaseLayer
    {
        public readonly MessageShape InputMessageShape;
        public readonly MessageShape OutputMessageShape;

        public BaseLayer(MessageShape inputMessageShape, MessageShape outputMessageShape)
        {
            OutputMessageShape = outputMessageShape;
            InputMessageShape = inputMessageShape;
        }
    }
}
