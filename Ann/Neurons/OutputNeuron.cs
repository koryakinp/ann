using System;
using System.Collections.Generic;
using System.Text;
using Ann.Activators;

namespace Ann.Neurons
{
    public class OutputNeuron : Neuron
    {
        public OutputNeuron(int layerIndex, int neuronIndex, IActivator activator, IWeightInitializer weightInitializer) 
            : base(layerIndex, neuronIndex, activator, weightInitializer) {}

        private double _target { get; set; }

        public void SetTarget(double value)
        {
            _target = value;
        }

        public override void CalculateError()
        {
            Error = 0.5 * Math.Pow((Value - _target), 2);
        }
    }
}
