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
            : base(config.MessageShape, new MessageShape(config.MessageShape.Size, config.MessageShape.Depth))
        {
            _activator = ActivatorFactory.Produce(config.ActivatorType);
            if (InputMessageShape.Depth == 1)
            {
                _cache = new double[InputMessageShape.Size];
            }
            else
            {
                _cache = new double[
                    InputMessageShape.Depth,
                    InputMessageShape.Size,
                    InputMessageShape.Size];
            }
        }

        public override LayerConfiguration GetLayerConfiguration()
        {
            return new ActivationLayerConfiguration(InputMessageShape, _activator.GetActivatorType());
        }

        public override Array PassBackward(Array input)
        {
            _cache.UpdateForEach<double>((q,idx) => 
                _activator.CalculateDeriviative(q) * (double)input.GetValue(idx));
            return _cache;
        }

        public override Array PassForward(Array input)
        {
            _cache.UpdateForEach<double>((q,idx) => (double)input.GetValue(idx));
            input.UpdateForEach<double>((q,idx) => _activator.CalculateValue((double)input.GetValue(idx)));
            return input;
        }
    }
}
