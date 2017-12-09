using Ann.Activators;
using Ann.Neurons;
using System.Collections.Generic;
using System.Linq;

namespace Ann.Layers
{
    public class InputLayer : Layer
    {
        public InputLayer(List<InputNeuron> neurons) : base(neurons) {}

        public void SetInputs(List<double> inputs)
        {
            foreach (InputNeuron neuron in Neurons.OrderBy(q => q.NeuronIndex))
            {
                neuron.SetInput(inputs[neuron.NeuronIndex - 1]);
            }
        }
    }
}
