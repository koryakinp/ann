using System.Collections.Generic;
using Ann.Connections;
using System;

namespace Ann.Neurons
{
    public class InputNeuron : Neuron
    {
        public InputNeuron(int neuronIndex) 
            : base(neuronIndex) {}

        private double _input { get; set; }

        public override void SetBackwardConnections(List<BackwardConnection> connections)
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
    }
}
