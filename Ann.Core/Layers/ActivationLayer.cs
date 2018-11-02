using Ann.Activators;
using Ann.Utils;
using System;

namespace Ann.Core.Layers
{
    public class ActivationLayer : Layer
    {
        private readonly Activators.Activator _activator;
        private readonly double[,,] _cache;

        public ActivationLayer(
            MessageShape inputMessageShape, 
            ActivatorType type) 
            : base(inputMessageShape, new MessageShape(inputMessageShape.Size, inputMessageShape.Depth))
        {
            _activator = ActivatorFactory.Produce(type);
            _cache = new double[
                InputMessageShape.Depth,
                InputMessageShape.Size, 
                InputMessageShape.Size];
        }

        public override Array PassBackward(Array input)
        {
            _cache.UpdateForEach<double>((q,idx) => 
                _activator.CalculateDeriviative(q) * (double)input.GetValue(idx));
            return _cache;
        }

        public override Array PassForward(Array input)
        {
            _cache.UpdateForEach<double>((q,idx) => 
                _activator.CalculateValue((double)input.GetValue(idx)));
            return _cache;
        }
    }
}
