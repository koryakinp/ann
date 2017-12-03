using Ann.Activators;
using Ann.Neurons;
using Ann.WeightInitializers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ann.Layers
{
    public class Layer
    {
        public readonly List<Neuron> Neurons;
        public readonly int LayerIndex;
        protected readonly IWeightInitializer _weightInitializer;

        public Layer(int numberOfNeurons, int layerIndex, IActivator activator, IWeightInitializer weightInitializer)
        {
            _weightInitializer = weightInitializer;
            LayerIndex = layerIndex;
            Neurons = new List<Neuron>();
            for (int i = 0; i < numberOfNeurons; i++)
            {
                Neurons.Add(Create(LayerIndex, i + 1, activator));
            }
        }

        public virtual void RandomizeWeights()
        {
            foreach (var neuron in Neurons)
            {
                neuron.RandomizeWeights();
            }
        }

        public virtual Neuron Create(int layerIndex, int neuronIndex, IActivator activator)
        {
            return new Neuron(layerIndex, neuronIndex, activator, _weightInitializer);
        }

        public void CalculateValue()
        {
            foreach (var neuron in Neurons)
            {
                neuron.CalculateValue();
            }
        }

        public void CalculateError()
        {
            foreach (var neuron in Neurons)
            {
                neuron.CalculateError();
            }
        }
    }
}
