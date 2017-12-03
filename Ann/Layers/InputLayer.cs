using Ann.Activators;
using Ann.Neurons;
using System.Collections.Generic;
using System.Linq;

namespace Ann.Layers
{
    public class InputLayer : Layer
    {
        public InputLayer(int numberOfNeurons, IWeightInitializer initializer) : base(numberOfNeurons, 1, new IdentityActivator(), null) {}

        public void SetInputs(List<double> inputs)
        {
            foreach (InputNeuron neuron in Neurons.OrderBy(q => q.NeuronIndex))
            {
                neuron.SetInput(inputs[neuron.NeuronIndex - 1]);
            }
        }

        public override Neuron Create(int layerIndex, int neuronIndex, IActivator activator)
        {
            return new InputNeuron(neuronIndex);
        }
    }
}
