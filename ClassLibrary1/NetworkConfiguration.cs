using Ann;
using Ann.Activators;
using Ann.Layers;
using Ann.Neurons;
using Ann.WeightInitializers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Ann
{
    public class NetworkConfiguration
    {
        public readonly LayerConfiguration LayerConfiguration;
        public readonly double LearningRate;
        public readonly double Momentum;

        public NetworkConfiguration(LayerConfiguration layerConfiguration, double learningRate = 0.05, double momentum = 0.9)
        {
            if(learningRate <= 0 || learningRate > 1)
            {

            }
            else if(momentum < 0 || momentum > 1)
            {

            }

            LayerConfiguration = layerConfiguration;
        }

    }

    public class LayerConfiguration
    {
        public readonly List<LayerConfigurationItem> Layers;

        public LayerConfiguration()
        {
            Layers = new List<LayerConfigurationItem>();
        }

        public LayerConfiguration AddInputLayer(int numberOfNeurons)
        {
            Layers.Add(new LayerConfigurationItem(LayerType.Input, numberOfNeurons));
            return this;
        }

        #region AddHiddenLayer overloads
        public LayerConfiguration AddHiddenLayer(int numberOfNeurons)
        {
            Layers.Add(new LayerConfigurationItem(
                LayerType.Hidden, 
                numberOfNeurons, 
                new LogisticActivator(),
                new DefaultWeightInitializer()));
            return this;
        }

        public LayerConfiguration AddHiddenLayer(int numberOfNeurons, IActivator activator) 
        {
            Layers.Add(new LayerConfigurationItem(
                LayerType.Hidden, 
                numberOfNeurons, 
                activator, 
                new DefaultWeightInitializer()));
            return this;
        }

        public LayerConfiguration AddHiddenLayer(int numberOfNeurons, ActivatorType activatorType)
        {
            Layers.Add(new LayerConfigurationItem(
                LayerType.Hidden, 
                numberOfNeurons, 
                ActivatorFactory.Produce(activatorType), 
                new DefaultWeightInitializer()));

            return this;
        }

        public LayerConfiguration AddHiddenLayer(int numberOfNeurons, IActivator activator, IWeightInitializer weightInitializer)
        {
            Layers.Add(new LayerConfigurationItem(
                LayerType.Hidden,
                numberOfNeurons,
                activator,
                weightInitializer));

            return this;
        }

        public LayerConfiguration AddHiddenLayer(int numberOfNeurons, ActivatorType activatorType, IWeightInitializer weightInitializer)
        {
            Layers.Add(new LayerConfigurationItem(
                LayerType.Hidden,
                numberOfNeurons,
                ActivatorFactory.Produce(activatorType),
                weightInitializer));

            return this;
        }
        #endregion

        #region AddOutputLayer overloads
        public LayerConfiguration AddOutputLayer(int numberOfNeurons)
        {
            Layers.Add(new LayerConfigurationItem(
                LayerType.Output,
                numberOfNeurons,
                new LogisticActivator(),
                new DefaultWeightInitializer()));
            return this;
        }

        public LayerConfiguration AddOutputLayer(int numberOfNeurons, IActivator activator)
        {
            Layers.Add(new LayerConfigurationItem(
                LayerType.Output, 
                numberOfNeurons, 
                activator, 
                new DefaultWeightInitializer()));
            return this;
        }

        public LayerConfiguration AddOutputLayer(int numberOfNeurons, ActivatorType activatorType)
        {
            Layers.Add(new LayerConfigurationItem(
                LayerType.Output, 
                numberOfNeurons, 
                ActivatorFactory.Produce(activatorType),
                new DefaultWeightInitializer()));
            return this;
        }

        public LayerConfiguration AddOutputLayer(int numberOfNeurons, IActivator activator, IWeightInitializer weightInitializer)
        {
            Layers.Add(new LayerConfigurationItem(
                LayerType.Output,
                numberOfNeurons,
                activator,
                weightInitializer));
            return this;
        }

        public LayerConfiguration AddOutputLayer(int numberOfNeurons, ActivatorType activatorType, IWeightInitializer weightInitializer)
        {
            Layers.Add(new LayerConfigurationItem(
                LayerType.Output,
                numberOfNeurons,
                ActivatorFactory.Produce(activatorType),
                weightInitializer));
            return this;
        }
        #endregion
    }

    public class LayerConfigurationItem
    {
        public readonly LayerType LayerType;
        public readonly int NumberOfNeurons;
        public readonly IActivator Activator;
        public readonly IWeightInitializer WeightInitializer;

        public LayerConfigurationItem(LayerType layerType, int numberOfNeurons, IActivator activator, IWeightInitializer weightInitializer)
        {
            WeightInitializer = weightInitializer;
            Activator = activator;
            LayerType = layerType;
            NumberOfNeurons = numberOfNeurons;
        }

        public LayerConfigurationItem(LayerType layerType, int numberOfNeurons)
        {
            LayerType = layerType;
            NumberOfNeurons = numberOfNeurons;
        }
    }

    public enum LayerType
    {
        Input,
        Hidden,
        Output
    }
}
