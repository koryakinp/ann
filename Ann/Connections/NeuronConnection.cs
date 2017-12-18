using Ann.Configuration;
using Ann.Neurons;

namespace Ann.Connections
{
    public class NeuronConnection
    {
        private readonly Connection _connection;
        private readonly Neuron _neuron;
        private double _velocity;

        public NeuronConnection(Connection connection, Neuron neuron)
        {
            _velocity = 0;
            _connection = connection;
            _neuron = neuron;
        }

        public double GetWeightedDelta()
        {
            return _connection.Weight * _neuron.Delta;
        }

        public double GetWeight()
        {
            return _connection.Weight;
        }

        public double GetWeightedValue()
        {
            return _connection.Weight * _neuron.Value;
        }

        public double GetValue()
        {
            return _neuron.Value;
        }

        public void UpdateWeight(double gradient, NetworkConfiguration meta)
        {
            _velocity = meta.Momentum * _velocity - meta.LearningRate * gradient;
            _connection.Weight += _velocity;
        }

        public void SetWeight(double weight)
        {
            _connection.Weight = weight;
        }
    }
}
