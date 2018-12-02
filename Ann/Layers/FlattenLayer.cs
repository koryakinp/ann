using Ann.Persistence;
using Ann.Persistence.LayerConfig;
using System;

namespace Ann.Layers
{
    internal class FlattenLayer : Layer
    {
        public FlattenLayer(FlattenLayerConfiguration config)
            : base(config.MessageShape, ComputeOutputMessageShape(config.MessageShape)) { }

        public override Array PassBackward(Array input)
        {
            var output = new double[InputMessageShape.Depth, InputMessageShape.Size, InputMessageShape.Size];

            int index = 0;

            for (int i = 0; i < InputMessageShape.Size; i++)
            {
                for (int j = 0; j < InputMessageShape.Size; j++)
                {
                    for (int k = 0; k < InputMessageShape.Depth; k++)
                    {
                        output[k, i, j] = (double)input.GetValue(index++);
                    }
                }
            }

            return output;
        }

        public override Array PassForward(Array input)
        {
            var output = new double[input.GetLength(0) * input.GetLength(1) * input.GetLength(2)];

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

        public override LayerConfiguration GetLayerConfiguration()
        {
            return new FlattenLayerConfiguration(InputMessageShape);
        }
    }
}
