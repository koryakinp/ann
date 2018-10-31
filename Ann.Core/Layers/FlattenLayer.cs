using Ann.Utils;
using System;
using System.Linq;

namespace Ann.Core.Layers
{
    public class FlattenLayer : Layer
    {
        public FlattenLayer(MessageShape inputMessageShape) 
            : base(inputMessageShape, ComputeOutputMessageShape(inputMessageShape)) {}

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
    }
}
