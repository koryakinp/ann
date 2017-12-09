using Ann.Neurons;

namespace Ann.Connections
{
    public class ForwardConnection
    {
        private readonly Connection _connection;
        private readonly Neuron _neuron;

        public ForwardConnection(Connection connection, Neuron neuron)
        {
            _connection = connection;
            _neuron = neuron;
        }

        public double GetWeightedDelta()
        {
            return _connection.Weight * _neuron.Delta;
        }
    }
}
