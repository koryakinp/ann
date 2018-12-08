using System;
using Ann.Utils;

namespace Ann.Layers.Pooling
{
    internal class PoolingFullLayer : PoolingForwardLayer, IFullLayer
    {
        private readonly bool[,,] _cache;

        public PoolingFullLayer(
            MessageShape inputMessageShape, int stride) 
            : base(inputMessageShape, stride)
        {
            _cache = new bool[
                inputMessageShape.Depth,
                inputMessageShape.Size,
                inputMessageShape.Size];
        }

        public new Array PassForward(Array input)
        {
            var res = MatrixHelper.MaxPool(input as double[,,], Stride);
            res.Cache.CopyTo<bool>(_cache);
            return res.Values;
        }

        public Array PassBackward(Array error)
        {
            double[,,] output = new double[_cache.GetLength(0), _cache.GetLength(1), _cache.GetLength(2)];

            output.ForEach((i, j, k) =>
            {
                output[i, j, k] = _cache[i, j, k]
                    ? (double)error.GetValue(i, j / Stride, k / Stride)
                    : output[i, j, k] = 0;
            });

            return output;
        }
    }
}
