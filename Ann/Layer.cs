using Activator = Ann.Activators.Activator;
using System.Collections.Generic;
using System.Linq;

namespace Ann
{
    internal class Layer
    {
        public readonly IReadOnlyList<Neuron> Neurons;
        private int _numberOfNeuronsInPreviouseLayer;
        public readonly int LayerIndex;

        public Layer(
            Activator activator,
            int numberOfNeurons,
            int numberOfNeuronsInPreviouseLayer,
            LearningRateAnnealerType lrat,
            int layerIndex)
        {
            LayerIndex = layerIndex;
            _numberOfNeuronsInPreviouseLayer = numberOfNeuronsInPreviouseLayer;

            List<Neuron> neurons = new List<Neuron>();
            for (int i = 0; i < numberOfNeurons; i++)
            {
                neurons.Add(new Neuron(activator, numberOfNeuronsInPreviouseLayer, lrat));
            }

            Neurons = new List<Neuron>(neurons);
        }

        public double[] PassForward(double[] value)
        {
            foreach (var neuron in Neurons)
            {
                double weightedSum = neuron
                    .Weights
                    .Select((w, j) => value[j] * w.Value)
                    .Sum();

                neuron.ComputeOutput(weightedSum);
            }

            return Neurons.Select(q => q.Output).ToArray();
        }

        public double[] PassBackward(double[] value)
        {
            Neurons.ForEach((q, i) => q.ComputeDelta(value[i]));
            double[] deltas = new double[_numberOfNeuronsInPreviouseLayer];

            foreach (var neuron in Neurons)
            {
                neuron
                    .Weights
                    .ForEach((q, i) => deltas[i] += q.Value * neuron.Delta);
            }

            return deltas;
        }

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
