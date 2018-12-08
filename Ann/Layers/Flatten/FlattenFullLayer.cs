using System;

namespace Ann.Layers.Flatten
{
    internal class FlattenFullLayer : FlattenForwardLayer, IFullLayer
    {
        public FlattenFullLayer(MessageShape inputMessageShape) : base(inputMessageShape) {}

        public Array PassBackward(Array error)
        {
            var shape = GetInputMessageShape();
            var output = new double[shape.Depth, shape.Size, shape.Size];

            int index = 0;

            for (int i = 0; i < shape.Size; i++)
            {
                for (int j = 0; j < shape.Size; j++)
                {
                    for (int k = 0; k < shape.Depth; k++)
                    {
                        output[k, i, j] = (double)error.GetValue(index++);
                    }
                }
            }

            return output;
        }
    }
}
