using System.Collections.Generic;
using Ann.Neurons;
using System.Linq;

namespace Ann.Layers
{
    public class OutputLayer : Layer
    {
        public OutputLayer(List<OutputNeuron> neurons, int layerIndex) : base(neurons, layerIndex) {}

        public void SetTarget(List<double> inputs)
        {
            foreach (OutputNeuron neuron in Neurons.OrderBy(q => q.NeuronIndex))
            {
                neuron.Target = inputs[neuron.NeuronIndex - 1];
            }
        }

        public double GetTotalError()
        {
            return Neurons
                .OfType<OutputNeuron>()
                .Sum(q => q.GetError());
        }

        public List<double> GetOutputValues()
        {
            return Neurons
                .OrderBy(q => q.NeuronIndex)
                .Select(q => q.Value)
                .ToList();
        }
    }
}
