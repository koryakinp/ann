using Ann.Neurons;

namespace Ann.Connections
{
    public class BackwardConnection
    {
        private readonly Connection _connection;
        private readonly Neuron _neuron;

        public BackwardConnection(Connection connection, Neuron neuron)
        {
            _connection = connection;
            _neuron = neuron;
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
            _connection.Weight = _connection.Weight - weight;
        }

        public void SetWeight(double weight)
        {
            _connection.Weight = weight;
        }
    }
}
