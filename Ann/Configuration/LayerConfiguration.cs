using Ann.Activators;
using Ann.WeightInitializers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ann.Configuration
{
    public class LayerConfiguration
    {
        internal readonly List<LayerConfigurationItem> Layers;

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
}
