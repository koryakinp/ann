using Ann.Utils;
using System;

namespace Ann.Core.Layers
{
    public class PoolingLayer : KernelLayer
    {
        private readonly int _stride;
        private bool[,,] _cache;

        public PoolingLayer(int kernelSize, MessageShape inputMessageShape) 
            : base(inputMessageShape, BuildOutputMessageShape(inputMessageShape, kernelSize))
        {
            _cache = new bool[
                InputMessageShape.Depth,
                InputMessageShape.Size, 
                InputMessageShape.Size];
            _stride = kernelSize;
        }

        public static MessageShape BuildOutputMessageShape(MessageShape inputMessageShape, int stride)
        {
            int size = inputMessageShape.Size % stride == 0
                ? inputMessageShape.Size / stride
                : (inputMessageShape.Size / stride) + 1;

            return new MessageShape(size, inputMessageShape.Depth);
        }

        public override Array PassBackward(Array input)
        {
            double[,,] output = new double[
                _cache.GetLength(0), 
                _cache.GetLength(1), 
                _cache.GetLength(2)];

            output.ForEach((i, j, k) =>
            {
                output[i, j, k] = _cache[i, j, k]
                    ? (double)input.GetValue(i, j / _stride, k / _stride)
                    : output[i, j, k] = 0;
            });

            return output;
        }

        public override Array PassForward(Array input)
        {
            var res = MatrixHelper.MaxPool(input as double[,,], _stride);
            _cache = res.Cache;
            return res.Values;
        }
    }
}
