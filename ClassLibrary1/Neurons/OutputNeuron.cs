using System;
using System.Collections.Generic;
using Ann.Connections;
using Ann.Activators;

namespace Ann.Neurons
{
    public class OutputNeuron : Neuron
    {
        public OutputNeuron(int neuronIndex, IActivator activator, IWeightInitializer weightInitializer) 
            : base(neuronIndex,activator,weightInitializer) {}

        public double Target { get; set; }

        public override void SetForwardConnections(List<ForwardConnection> connections)
        {
            throw new NotSupportedException();
        }

        public override void CalculateDelta()
        {
            Delta = Value - Target;
        }

        public double GetError()
        {
            return 0.5 * Math.Pow((Value - Target), 2);
        }
    }
}
