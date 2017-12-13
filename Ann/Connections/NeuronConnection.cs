using Ann.Neurons;

namespace Ann.Connections
{
    public class NeuronConnection
    {
        private readonly Connection _connection;
        private readonly Neuron _neuron;

        public NeuronConnection(Connection connection, Neuron neuron)
        {
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

        public void UpdateWeight(double weight)
        {
            _connection.Weight -= weight;
        }

        public void SetWeight(double weight)
        {
            _connection.Weight = weight;
        }
    }
}
