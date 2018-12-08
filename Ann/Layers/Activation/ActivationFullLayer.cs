using System;
using Ann.Activators;
using Ann.Utils;

namespace Ann.Layers.Activation
{
    internal class ActivationFullLayer : ActivationForwardLayer, IFullLayer
    {
        private readonly Array _cache;

        public ActivationFullLayer(
            ActivatorType type, 
            MessageShape inputMessageShape) 
            : base(type, inputMessageShape)
        {
            _cache = inputMessageShape.Depth == 1
                ? _cache = new double[inputMessageShape.Size]
                : _cache = new double[inputMessageShape.Depth, 
                    inputMessageShape.Size, 
                    inputMessageShape.Size];
        }

        public Array PassBackward(Array error)
        {
            _cache.UpdateForEach<double>((q, idx) =>
                Activator.CalculateDeriviative(q) * (double)error.GetValue(idx));
            return _cache;
        }

        public new Array PassForward(Array input)
        {
            var output = base.PassForward(input);
            output.CopyTo<double>(_cache);
            return output;
        }
    }
}
