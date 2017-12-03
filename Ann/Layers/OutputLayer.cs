using Ann.Activators;
using System;
using System.Collections.Generic;
using System.Text;
using Ann.Neurons;
using System.Linq;

namespace Ann.Layers
{
    public class OutputLayer : Layer
    {
        public OutputLayer(int numberOfNeurons, int layerIndex, IActivator activator, IWeightInitializer weightInitializer) 
            : base(numberOfNeurons, layerIndex, activator, weightInitializer) {}

        public void SetTarget(List<double> inputs)
        {
            foreach (OutputNeuron neuron in Neurons.OrderBy(q => q.NeuronIndex))
            {
                neuron.SetTarget(inputs[neuron.NeuronIndex - 1]);
            }
        }

        public override Neuron Create(int layerIndex, int neuronIndex, IActivator activator)
        {
            return new OutputNeuron(layerIndex, neuronIndex, activator, _weightInitializer);
        }

        public double GetTotalError()
        {
            return Neurons.Sum(q => q.Error);
        }
    }
}
