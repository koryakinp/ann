using Ann.Activators;
using Ann.WeightInitializers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;

namespace Ann.Neurons
{
    public class Neuron
    {
        private readonly List<Connection> _forwardConnections;
        private readonly List<Connection> _backwardConnections;
        public double Value { get; protected set; }
        public double Error { get; protected set; }
        private readonly IActivator _activator;
        private readonly IWeightInitializer _weightInitializer;

        public readonly int LayerIndex;
        public readonly int NeuronIndex;

        public Neuron(int layerIndex, int neuronIndex, IActivator activator, IWeightInitializer weightInitializer)
        {
            LayerIndex = layerIndex;
            NeuronIndex = neuronIndex;
            _forwardConnections = new List<Connection>();
            _backwardConnections = new List<Connection>();
            _activator = activator;
            _weightInitializer = weightInitializer;
        }

        public void SetUpForwardConnections(IEnumerable<Connection> connections)
        {
            _forwardConnections.AddRange(connections);
        }

        public void SetUpBackwardConnections(IEnumerable<Connection> connections)
        {
            _backwardConnections.AddRange(connections);
        }

        public virtual void CalculateValue()
        {
            double weightedSum = _backwardConnections.Sum(q => q.Weight * q.BackwardNeuron.Value);
            Value = _activator.CalculateValue(weightedSum);
        }

        public virtual void CalculateError()
        {
            throw new NotImplementedException();
        }

        public void RandomizeWeights()
        {
            int numberOfInputs = _forwardConnections.Count;
            int numberOfOutputs = _backwardConnections.Count;

            foreach (var connection in _forwardConnections)
            {
                connection.SetNewWeight(_weightInitializer.InitializeWeight(numberOfInputs, numberOfOutputs));
                connection.UpdateWeight();
            }
        }
    }
}
