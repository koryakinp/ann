using Ann.Activators;
using Ann.WeightInitializers;
using System.Collections.Generic;

namespace Ann.Configuration
{
    public class LayerConfiguration
    {
        internal readonly List<LayerConfigurationItem> Layers;

        /// <summary>
        /// Creates new LayerConfiguration object
        /// </summary>
        public LayerConfiguration()
        {
            Layers = new List<LayerConfigurationItem>();
        }

        /// <summary>
        /// Adds an input layer to a Layer Configuration with specified number of input neurons
        /// </summary>
        /// <param name="numberOfNeurons">Number of inputs in the layer</param>
        /// <returns>Layer Configuration object</returns>
        public LayerConfiguration AddInputLayer(int numberOfNeurons)
        {
            Layers.Add(new LayerConfigurationItem(LayerType.Input, numberOfNeurons));
            return this;
        }

        #region AddHiddenLayer overloads
        /// <summary>
        /// Adds a hidden layer to a Layer Configuration with specified number of neurons,
        /// logistic activator and default weight initializer
        /// </summary>
        /// <param name="numberOfNeurons">Number of neurons in the layer</param>
        /// <returns>Layer Configuration object</returns>
        public LayerConfiguration AddHiddenLayer(int numberOfNeurons)
        {
            Layers.Add(new LayerConfigurationItem(
                LayerType.Hidden,
                numberOfNeurons,
                new LogisticActivator(),
                new DefaultWeightInitializer()));
            return this;
        }

        /// <summary>
        /// Adds a hidden layer to a Layer Configuration with specified number of neurons, 
        /// custom neuron activator and default weight initializer
        /// </summary>
        /// <param name="numberOfNeurons">Number of neurons in the layer</param>
        /// <param name="activator">Custom neuron activator</param>
        /// <returns>Layer Configuration object</returns>
        public LayerConfiguration AddHiddenLayer(int numberOfNeurons, IActivator activator)
        {
            Layers.Add(new LayerConfigurationItem(
                LayerType.Hidden,
                numberOfNeurons,
                activator,
                new DefaultWeightInitializer()));
            return this;
        }

        /// <summary>
        /// Adds a hidden layer to a Layer Configuration with specified number of neurons,
        /// predefined activator type and default weight initializer
        /// </summary>
        /// <param name="numberOfNeurons">Number of neurons in the layer</param>
        /// <param name="activatorType">Activator type</param>
        /// <returns>Layer Configuration object</returns>
        public LayerConfiguration AddHiddenLayer(int numberOfNeurons, ActivatorType activatorType)
        {
            Layers.Add(new LayerConfigurationItem(
                LayerType.Hidden,
                numberOfNeurons,
                ActivatorFactory.Produce(activatorType),
                new DefaultWeightInitializer()));

            return this;
        }

        /// <summary>
        /// Adds a hidden layer to a Layer Configuration with specified number of neurons, 
        /// custom neuron activator and custom weight initializer 
        /// </summary>
        /// <param name="numberOfNeurons">Number of neurons in the layer</param>
        /// <param name="activator">Custom neuron activator</param>
        /// <param name="weightInitializer">Custom weight initializer</param>
        /// <returns>Layer Configuration object</returns>
        public LayerConfiguration AddHiddenLayer(int numberOfNeurons, IActivator activator, IWeightInitializer weightInitializer)
        {
            Layers.Add(new LayerConfigurationItem(
                LayerType.Hidden,
                numberOfNeurons,
                activator,
                weightInitializer));

            return this;
        }

        /// <summary>
        /// Adds a hidden layer to a Layer Configuration with specified number of neurons, 
        /// predefined activator type and custom weight initializer 
        /// </summary>
        /// <param name="numberOfNeurons">Number of neurons in the layer</param>
        /// <param name="activatorType">Activator type</param>
        /// <param name="weightInitializer">Custom weight initializer</param>
        /// <returns>Layer Configuration object</returns>
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
        /// <summary>
        /// Adds an output layer to a Layer Configuration with specified number of input neurons, 
        /// logistic neuron activator and default weight initializer
        /// </summary>
        /// <param name="numberOfNeurons">Number of neurons in the layer</param>
        /// <returns>Layer Configuration object</returns>
        public LayerConfiguration AddOutputLayer(int numberOfNeurons)
        {
            Layers.Add(new LayerConfigurationItem(
                LayerType.Output,
                numberOfNeurons,
                new LogisticActivator(),
                new DefaultWeightInitializer()));
            return this;
        }


        /// <summary>
        /// Adds an output layer to a Layer Configuration with specified number of neurons, 
        /// custom activator and default weight initializer 
        /// </summary>
        /// <param name="numberOfNeurons">Number of neurons in the layer</param>
        /// <param name="activator">Custom neuron activator</param>
        /// <returns>Layer Configuration object</returns>
        public LayerConfiguration AddOutputLayer(int numberOfNeurons, IActivator activator)
        {
            Layers.Add(new LayerConfigurationItem(
                LayerType.Output,
                numberOfNeurons,
                activator,
                new DefaultWeightInitializer()));
            return this;
        }

        /// <summary>
        /// Adds an output layer to a Layer Configuration with specified number of neurons,
        /// predefined activator type and default weight initializer
        /// </summary>
        /// <param name="numberOfNeurons">Number of neurons in the layer</param>
        /// <param name="activatorType">Activator type</param>
        /// <returns>Layer Configuration object</returns>
        public LayerConfiguration AddOutputLayer(int numberOfNeurons, ActivatorType activatorType)
        {
            Layers.Add(new LayerConfigurationItem(
                LayerType.Output,
                numberOfNeurons,
                ActivatorFactory.Produce(activatorType),
                new DefaultWeightInitializer()));
            return this;
        }

        /// <summary>
        /// Adds an output layer to a Layer Configuration with specified number of neurons, 
        /// custom neuron activator and custom weight initializer 
        /// </summary>
        /// <param name="numberOfNeurons">Number of neurons in the layer</param>
        /// <param name="activator">Custom neuron activator</param>
        /// <param name="weightInitializer">Custom weight initializer</param>
        /// <returns>Layer Configuration object</returns>
        public LayerConfiguration AddOutputLayer(int numberOfNeurons, IActivator activator, IWeightInitializer weightInitializer)
        {
            Layers.Add(new LayerConfigurationItem(
                LayerType.Output,
                numberOfNeurons,
                activator,
                weightInitializer));
            return this;
        }

        /// <summary>
        /// Adds an output layer to a Layer Configuration with specified number of neurons, 
        /// predefined activator type and custom weight initializer 
        /// </summary>
        /// <param name="numberOfNeurons">Number of neurons in the layer</param>
        /// <param name="activatorType">Activator type</param>
        /// <param name="weightInitializer">Custom weight initializer</param>
        /// <returns>Layer Configuration object</returns>
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
