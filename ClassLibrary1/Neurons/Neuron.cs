using Ann.Activators;
using Ann.Connections;
using System.Collections.Generic;
using System.Linq;

namespace Ann.Neurons
{
    public class Neuron
    {
        protected readonly List<ForwardConnection> ForwardConnections;
        protected readonly List<BackwardConnection> BackwardConnections;
        public double Value { get; protected set; }
        public double Delta { get; protected set; }
        public double Bias { get; protected set; }
        protected readonly IActivator Activator;
        private readonly IWeightInitializer _weightInitializer;
        public readonly int NeuronIndex;

        public Neuron(
            int neuronIndex,
            IActivator activator, 
            IWeightInitializer weightInitializer)
        {
            NeuronIndex = neuronIndex;
            ForwardConnections = new List<ForwardConnection>();
            BackwardConnections = new List<BackwardConnection>();
            Activator = activator;
            _weightInitializer = weightInitializer;
        }

        protected Neuron(int neuronIndex)
        {
            NeuronIndex = neuronIndex;
            ForwardConnections = new List<ForwardConnection>();
            BackwardConnections = new List<BackwardConnection>();
        }

        public virtual void SetForwardConnections(List<ForwardConnection> connections)
        {
            ForwardConnections.AddRange(connections);
        }

        public virtual void SetBackwardConnections(List<BackwardConnection> connections)
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

        public void UpdateWeights(NetworkMeta meta)
        {
            foreach (var connection in BackwardConnections)
            {
                double value = Delta * connection.GetValue();
                connection.UpdateWeight(meta.LearningRate * value);
            }

            Bias = Bias - Delta * meta.LearningRate;
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
    }
}
