using System;
using Ann.Persistence;
using Ann.Persistence.LayerConfig;

namespace Ann.Layers.Pooling
{
    class PoolingForwardLayer : BaseLayer, IForwardLayer
    {
        protected readonly int Stride;

        public PoolingForwardLayer(MessageShape inputMessageShape, int stride) 
            : base(inputMessageShape, BuildOutputMessageShape(inputMessageShape, stride))
        {
            Stride = stride;
        }

        public Array PassForward(Array input)
        {
            return MatrixHelper.MaxPool(input as double[,,], Stride).Values;
        }

        public static MessageShape BuildOutputMessageShape(MessageShape inputMessageShape, int stride)
        {
            int size = inputMessageShape.Size % stride == 0
                ? inputMessageShape.Size / stride
                : (inputMessageShape.Size / stride) + 1;

            return new MessageShape(size, inputMessageShape.Depth);
        }

        public LayerConfiguration GetConfiguration()
        {
            return new PoolingLayerConfiguration(Stride, GetInputMessageShape());
        }
    }
}
