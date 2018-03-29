using System;
using System.Linq;

namespace Ann.Layers
{
    internal class OuputLayer : Layer
    {
        public OuputLayer(
            int numberOfNeurons, 
            int numberOfNeuronsInPreviouseLayer, 
            LearningRateAnnealerType lrat, 
            int layerIndex) 
            : base(numberOfNeurons, 
                  numberOfNeuronsInPreviouseLayer, 
                  lrat, 
                  layerIndex) {}

        public override double[] PassBackward(double[] value)
        {
            double[] deltas = new double[NumberOfNeuronsInPreviouseLayer];
            Neurons.ForEach((q, i) => q.Delta = value[i] * Neurons[i].Output * (1 - Neurons[i].Output));
            Neurons.ForEach(q => q.Weights.ForEach((w, i) => deltas[i] += w.Value * q.Delta));
            return deltas;
        }

        public override double[] PassForward(double[] value)
        {
            var temp = new double[Neurons.Count];

            Neurons.ForEach((q,i) =>
            {
                temp[i] = Math.Exp(q.Weights.Select((w, j) => value[j] * w.Value).Sum() + q.Bias.Value);
            });

            var sum = temp.Sum();

            Neurons.ForEach((q, i) => q.Output = temp[i] / sum);
            return Neurons.Select(q => q.Output).ToArray();
        }
    }
}
