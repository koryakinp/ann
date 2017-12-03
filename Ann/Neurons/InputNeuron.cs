using Ann.Activators;

namespace Ann.Neurons
{
    public class InputNeuron : Neuron
    {
        public InputNeuron(int neuronIndex) 
            : base(1, neuronIndex, new IdentityActivator(), null) {}

        public void SetInput(double input)
        {
            Value = input;
        }
    }
}
