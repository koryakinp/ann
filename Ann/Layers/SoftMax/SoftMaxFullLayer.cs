using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;

namespace Ann.Layers.SoftMax
{
    class SoftMaxFullLayer : SoftMaxForwardLayer, IFullLayer
    {
        private readonly Vector<double> _cache;

        public SoftMaxFullLayer(MessageShape inputMessageShape) 
            : base(inputMessageShape)
        {
            _cache = Vector.Build.Dense(inputMessageShape.Size);
        }

        public Array PassBackward(Array errors)
        {
            var E = errors as double[];
            return Activator.CalculateDeriviative(_cache.ToArray(), E);
        }

        public new Array PassForward(Array input)
        {
            _cache.SetValues(base.PassForward(input) as double[]);
            return _cache.ToArray();
        }
    }
}
