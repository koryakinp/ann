using Ann.Layers;
using Ann.Persistence;
using Ann.Persistence.LayerConfig;
using System;

namespace Ann
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
                    var layer1 = new ConvolutionLayer(c);
                    layer1.SetWeights(c.Weights);
                    layer1.SetBiases(c.Biases);
                    return layer1;
                case FlattenLayerConfiguration c:
                    return new FlattenLayer(c);
                case HiddenLayerConfiguration c:
                    var layer2 = new HiddenLayer(c);
                    layer2.SetWeights(c.Weights);
                    layer2.SetBiases(c.Biases);
                    return layer2;
                case InputLayerConfiguration c:
                    return new InputLayer(c);
                case PoolingLayerConfiguration c:
                    return new PoolingLayer(c);
                case SoftmaxLayerConfiguration c:
                    var layer3 = new SoftMaxLayer(c);
                    layer3.SetWeights(c.Weights);
                    return layer3;
                default:
                    throw new Exception("Config type is not supported");
            }
        }
    }
}
