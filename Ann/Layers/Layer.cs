using System.Collections.Generic;

namespace Ann.Layers
{
    internal abstract class Layer
    {
        public readonly IReadOnlyList<Neuron> Neurons;
        protected int NumberOfNeuronsInPreviouseLayer;
        public readonly int LayerIndex;

        public Layer(
            int numberOfNeurons,
            int numberOfNeuronsInPreviouseLayer,
            LearningRateAnnealerType lrat,
            int layerIndex)
        {
            LayerIndex = layerIndex;
            NumberOfNeuronsInPreviouseLayer = numberOfNeuronsInPreviouseLayer;

            List<Neuron> neurons = new List<Neuron>();
            for (int i = 0; i < numberOfNeurons; i++)
            {
                neurons.Add(new Neuron(numberOfNeuronsInPreviouseLayer, lrat));
            }

            Neurons = new List<Neuron>(neurons);
        }

        public abstract double[] PassForward(double[] value);

        public abstract double[] PassBackward(double[] value);

        public void UpdateWeights(double[] values)
        {
            Neurons.ForEach(q => q.UpdateWeights(values));
        }

        public void UpdateBiases()
        {
            Neurons.ForEach(q => q.UpdateBias());
        }

        public void RandomizeWeights()
        {
            Neurons.ForEach(q => q.RandomizeWeights());
        }
    }
}
