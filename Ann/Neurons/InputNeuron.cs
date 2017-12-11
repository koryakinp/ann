using System.Collections.Generic;
using Ann.Connections;
using System;
using Ann.Model;
using System.Linq;

namespace Ann.Neurons
{
    public class InputNeuron : Neuron
    {
        public InputNeuron(int neuronIndex) 
            : base(neuronIndex) {}

        private double _input { get; set; }

        public override void SetBackwardConnections(List<NeuronConnection> connections)
        {
            throw new NotSupportedException();
        }

        public void SetInput(double input)
        {
            _input = input;
        }

        public override void CalculateValue()
        {
            Value = _input;
        }

        internal override NeuronModel ToNeuronModel()
        {
            return new NeuronModel
            {
                Bias = Bias,
                NeuronIndex = NeuronIndex,
                Activator = string.Empty,
                Weights = ForwardConnections
                    .Select(q => q.GetWeight())
                    .ToList()
            };
        }
    }
}
