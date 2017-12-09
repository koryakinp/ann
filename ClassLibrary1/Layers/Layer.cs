using Ann.Neurons;
using System.Collections.Generic;

namespace Ann.Layers
{
    public class Layer
    {
        public readonly List<Neuron> Neurons;
        public readonly int LayerIndex;

        public Layer(List<Neuron> neurons, int layerIndex)
        {
            LayerIndex = layerIndex;
            Neurons = neurons;
        }

        protected Layer(List<InputNeuron> neurons)
        {
            LayerIndex = 1;
            Neurons = new List<Neuron>();
            foreach (var neuron in neurons)
            {
                Neurons.Add(neuron);
            }
        }

        protected Layer(List<OutputNeuron> neurons, int layerIndex)
        {
            LayerIndex = layerIndex;
            Neurons = new List<Neuron>();
            foreach (var neuron in neurons)
            {
                Neurons.Add(neuron);
            }
        }


        public virtual void RandomizeWeights()
        {
            foreach (var neuron in Neurons)
            {
                neuron.RandomizeWeights();
            }
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
                neuron.CalculateDelta();
            }
        }
    }
}
