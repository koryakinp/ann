using Ann.Utils;
using System;

namespace Ann.Core.Layers
{
    public class MaxPoolingLayer : Layer
    {
        private readonly int _stride;
        private bool[,,] _cache;

        public MaxPoolingLayer(int kernelSize, MessageShape inputMessageShape) 
            : base(inputMessageShape)
        {
            _cache = new bool[
                InputMessageShape.Size, 
                InputMessageShape.Size, 
                InputMessageShape.Depth];
            _stride = kernelSize;
        }

        public override MessageShape GetOutputMessageShape()
        {
            int size = InputMessageShape.Size % _stride == 0
                ? InputMessageShape.Size / _stride
                : (InputMessageShape.Size / _stride) + 1;

            return new MessageShape(size, InputMessageShape.Depth);
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
