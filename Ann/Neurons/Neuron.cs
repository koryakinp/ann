using Ann.Activators;
using Ann.Configuration;
using Ann.Connections;
using Ann.Model;
using System.Collections.Generic;
using System.Linq;

namespace Ann.Neurons
{
    public class Neuron
    {
        internal readonly List<NeuronConnection> ForwardConnections;
        internal readonly List<NeuronConnection> BackwardConnections;
        public double Value { get; protected set; }
        public double Delta { get; protected set; }
        public double Bias { get; protected set; }
        internal readonly IActivator Activator;
        private readonly IWeightInitializer _weightInitializer;
        public readonly int NeuronIndex;
        private double _velocity;

        public Neuron(
            int neuronIndex,
            IActivator activator, 
            IWeightInitializer weightInitializer)
        {
            _velocity = 0;
            NeuronIndex = neuronIndex;
            ForwardConnections = new List<NeuronConnection>();
            BackwardConnections = new List<NeuronConnection>();
            Activator = activator;
            _weightInitializer = weightInitializer;
        }

        protected Neuron(int neuronIndex)
        {
            NeuronIndex = neuronIndex;
            ForwardConnections = new List<NeuronConnection>();
            BackwardConnections = new List<NeuronConnection>();
        }

        public virtual void SetForwardConnections(List<NeuronConnection> connections)
        {
            ForwardConnections.AddRange(connections);
        }

        public virtual void SetBackwardConnections(List<NeuronConnection> connections)
        {
            BackwardConnections.AddRange(connections);
        }

        public virtual void CalculateValue()
        {
            double sum = BackwardConnections.Sum(q => q.GetWeightedValue()) + Bias;
            Value = Activator.CalculateValue(sum);
        }

        public virtual void CalculateDelta()
        {
            double sum = ForwardConnections.Sum(q => q.GetWeightedDelta());
            Delta = sum * Activator.CalculateDeriviative(Value);
        }

        public void UpdateWeights(NetworkConfiguration meta)
        {
            foreach (var connection in BackwardConnections)
            {

                double value = Delta * connection.GetValue();
                connection.UpdateWeight(value, meta);
            }

            _velocity = meta.Momentum * _velocity - Delta * meta.LearningRate;
            Bias += _velocity;
        }

        public void RandomizeWeights()
        {
            int numberOfOutputs = ForwardConnections.Count;
            int numberOfInputs = BackwardConnections.Count;

            foreach (var connection in BackwardConnections)
            {
                double weight = _weightInitializer
                    .InitializeWeight(numberOfInputs, numberOfOutputs);
                connection.SetWeight(weight);
            }
        }

        internal virtual NeuronModel ToNeuronModel()
        {
            return new NeuronModel
            {
                Bias = Bias,
                NeuronIndex = NeuronIndex,
                Activator = Activator.GetType().AssemblyQualifiedName,
                Weights = ForwardConnections
                    .Select(q => q.GetWeight())
                    .ToList()
            };
        }

        internal void SetBias(double bias)
        {
            Bias = bias;
        }
    }
}
