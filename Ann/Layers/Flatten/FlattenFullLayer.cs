using System;
using System.Collections.Generic;
using System.Text;

namespace Ann.Layers.Flatten
{
    class FlattenFullLayer : FlattenForwardLayer, IFullLayer
    {
        public FlattenFullLayer(MessageShape inputMessageShape) : base(inputMessageShape) {}

        public Array PassBackward(Array error)
        {
            var output = new double[InputMessageShape.Depth, InputMessageShape.Size, InputMessageShape.Size];

            int index = 0;

            for (int i = 0; i < InputMessageShape.Size; i++)
            {
                for (int j = 0; j < InputMessageShape.Size; j++)
                {
                    for (int k = 0; k < InputMessageShape.Depth; k++)
                    {
                        output[k, i, j] = (double)error.GetValue(index++);
                    }
                }
            }

            return output;
        }
    }
}
