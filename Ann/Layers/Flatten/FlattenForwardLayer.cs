using System;
using Ann.Persistence;
using Ann.Persistence.LayerConfig;

namespace Ann.Layers.Flatten
{
    internal class FlattenForwardLayer : BaseLayer, IForwardLayer
    {
        public FlattenForwardLayer(MessageShape inputMessageShape) 
            : base(inputMessageShape, ComputeOutputMessageShape(inputMessageShape)) {}

        public Array PassForward(Array input)
        {
            var output = new double[input.Length];

            int index = 0;

            for (int i = 0; i < input.GetLength(1); i++)
            {
                for (int j = 0; j < input.GetLength(2); j++)
                {
                    for (int k = 0; k < input.GetLength(0); k++)
                    {
                        output[index++] = (double)input.GetValue(k, i, j);
                    }
                }
            }

            return output;
        }

        public static MessageShape ComputeOutputMessageShape(MessageShape shape)
        {
            return new MessageShape(shape.Size * shape.Size * shape.Depth);
        }

        public LayerConfiguration GetConfiguration()
        {
            return new FlattenLayerConfiguration(GetInputMessageShape());
        }
    }
}
