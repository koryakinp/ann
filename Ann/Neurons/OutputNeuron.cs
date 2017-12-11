using System;
using System.Collections.Generic;
using Ann.Connections;
using Ann.Activators;
using Ann.Model;

namespace Ann.Neurons
{
    public class OutputNeuron : Neuron
    {
        public OutputNeuron(int neuronIndex, IActivator activator, IWeightInitializer weightInitializer) 
            : base(neuronIndex,activator,weightInitializer) {}

        public double Target { get; set; }

        public override void SetForwardConnections(List<NeuronConnection> connections)
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

        internal override NeuronModel ToNeuronModel()
        {
            return new NeuronModel
            {
                Bias = Bias,
                NeuronIndex = NeuronIndex,
                Activator = Activator.GetType().AssemblyQualifiedName,
                Weights = new List<double>()
            };
        }
    }
}
