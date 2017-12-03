using Ann.Neurons;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ann
{
    public class Connection
    {
        public readonly Neuron ForwardNeuron;
        public readonly Neuron BackwardNeuron;

        public double Weight { get; private set; }
        private double _newWeight { get; set; }

        public Connection(Neuron forwardNeuron, Neuron backwardNeuron)
        {
            ForwardNeuron = forwardNeuron;
            BackwardNeuron = backwardNeuron;
        }

        public void UpdateWeight()
        {
            Weight = _newWeight; 
        }

        public void SetNewWeight(double weight)
        {
            _newWeight = weight;
        }

        public override string ToString()
        {
            return $"Between {BackwardNeuron.LayerIndex}-{BackwardNeuron.NeuronIndex} and {ForwardNeuron.LayerIndex}-{ForwardNeuron.NeuronIndex}";
        }
    }
}
