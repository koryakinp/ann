using Ann.Layers;
using Ann.Layers.Activation;
using Ann.Layers.Convolution;
using Ann.Layers.Dense;
using Ann.Layers.Flatten;
using Ann.Layers.Input;
using Ann.Layers.Pooling;
using Ann.Layers.SoftMax;
using Ann.Persistence;
using Ann.Persistence.LayerConfig;
using System;

namespace Ann
{
    internal static class LayerFactory
    {
        public static IForwardLayer Produce(LayerConfiguration config)
        {
            switch (config)
            {
                case ActivationLayerConfiguration c:
                    return new ActivationForwardLayer(c.ActivatorType, c.MessageShape);
                case ConvolutionLayerConfigurtion c:
                    var layer1 = new ConvolutionForwardLayer(
                        c.MessageShape, c.KernelSize, c.NumberOfKernels);
                    layer1.SetWeights(c.Weights);
                    layer1.SetBiases(c.Biases);
                    return layer1;
                case FlattenLayerConfiguration c:
                    return new FlattenForwardLayer(c.MessageShape);
                case DenseLayerConfiguration c:
                    var layer2 = new DenseForwardLayer(
                        c.MessageShape, c.NumberOfNeurons, c.EnableBias);
                    layer2.SetWeights(c.Weights);
                    layer2.SetBiases(c.Biases);
                    return layer2;
                case InputLayerConfiguration c:
                    return new InputForwardLayer(c.MessageShape);
                case PoolingLayerConfiguration c:
                    return new PoolingForwardLayer(c.MessageShape, c.KernelSize);
                case SoftmaxLayerConfiguration c:
                    var layer3 = new SoftMaxForwardLayer(c.MessageShape);
                    return layer3;
                default:
                    throw new Exception("Config type is not supported");
            }
        }
    }
}
