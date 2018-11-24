using Ann.Persistence;
using Ann.Persistence.LayerConfig;
using Ann.Utils;
using System;
using System.Linq;

namespace Ann.Layers
{
    internal class FlattenLayer : Layer
    {
        public FlattenLayer(FlattenLayerConfiguration config)
            : base(config.MessageShape, ComputeOutputMessageShape(config.MessageShape)) { }

        public override Array PassBackward(Array input)
        {
            return ArrayConverter.Convert1Dto3D(
                input as double[], 
                new int[3] 
                {
                    InputMessageShape.Depth,
                    InputMessageShape.Size,
                    InputMessageShape.Size
                });
        }

        public override Array PassForward(Array input)
        {
            return input.OfType<double>().ToArray();
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
