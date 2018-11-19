using Ann.Core.Layers;
using Ann.Core.Persistence;
using Ann.Core.Persistence.LayerConfig;
using System;

namespace Ann.Core
{
    public static class LayerFactory
    {
        public static Layer Produce(LayerConfiguration config)
        {
            switch (config)
            {
                case ActivationLayerConfiguration c:
                    return new ActivationLayer(c);
                case ConvolutionLayerConfigurtion c:
                    return new ConvolutionLayer(c);
                case FlattenLayerConfiguration c:
                    return new FlattenLayer(c);
                case HiddenLayerConfiguration c:
                    return new HiddenLayer(c);
                case InputLayerConfiguration c:
                    return new InputLayer(c);
                case PoolingLayerConfiguration c:
                    return new PoolingLayer(c);
                case SoftmaxLayerConfiguration c:
                    return new SoftMaxLayer(c);
                default:
                    throw new Exception("Config type is not supported");
            }
        }
    }
}
