using Ann.Activators;
using Ann.Persistence;
using Ann.Persistence.LayerConfig;
using Ann.Utils;
using System;

namespace Ann.Layers
{
    internal class ActivationLayer : Layer
    {
        private readonly Activators.Activator _activator;
        private readonly Array _cache;

        public ActivationLayer(ActivationLayerConfiguration config)
            : base(config.MessageShape, config.MessageShape)
        {
            _activator = ActivatorFactory.Produce(config.ActivatorType);
            if (config.MessageShape.Depth == 1)
            {
                _cache = new double[config.MessageShape.Size];
            }
            else
            {
                _cache = new double[
                    config.MessageShape.Depth,
                    config.MessageShape.Size,
                    config.MessageShape.Size];
            }
        }

        public override LayerConfiguration GetLayerConfiguration()
        {
            return new ActivationLayerConfiguration(InputMessageShape, _activator.GetActivatorType());
        }

        public override Array PassBackward(Array input)
        {
            _cache.UpdateForEach<double>((q, idx) =>
                _activator.CalculateDeriviative(q) * (double)input.GetValue(idx));
            return _cache;
        }

        public override Array PassForward(Array input)
        {
            input.UpdateForEach<double>((q, idx) =>
            {
                _cache.SetValue(q, idx);
                var val = _activator.CalculateValue(q);
                return val;
            });
            return input;
        }
    }
}