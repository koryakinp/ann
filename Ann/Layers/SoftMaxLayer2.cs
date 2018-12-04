using Ann.Activators;
using Ann.Persistence;
using Ann.Persistence.LayerConfig;
using System;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra;

namespace Ann.Layers
{
    internal class SoftMaxLayer2 : Layer
    {
        private readonly SoftmaxActivator sofmaxActivator = new SoftmaxActivator();
        private readonly Vector<double> _cache;

        public SoftMaxLayer2(SoftmaxLayer2Configuration config)
            : base(config.MessageShape, config.MessageShape)
        {
            _cache = Vector.Build.Dense(config.MessageShape.Size);
        }

        public override LayerConfiguration GetLayerConfiguration()
        {
            return new SoftmaxLayer2Configuration(InputMessageShape);
        }

        public override Array PassBackward(Array errors)
        {
            var E = errors as double[];
            return sofmaxActivator.CalculateDeriviative(_cache.ToArray(), E);
        }

        public override Array PassForward(Array input)
        {
            _cache.SetValues(sofmaxActivator.CalculateValue(input as double[]));
            return _cache.ToArray();
        }
    }
}