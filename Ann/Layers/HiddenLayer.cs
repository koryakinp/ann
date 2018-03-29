using Activator = Ann.Activators.Activator;
using System.Linq;

namespace Ann.Layers
{
    internal class HiddenLayer : Layer
    {
        private readonly Activator _activator;

        public HiddenLayer(
            Activator activator,
            int numberOfNeurons,
            int numberOfNeuronsInPreviouseLayer,
            LearningRateAnnealerType lrat,
            int layerIndex) : base(numberOfNeurons, numberOfNeuronsInPreviouseLayer, lrat, layerIndex)
        {
            _activator = activator;
        }

        public override double[] PassForward(double[] value)
        {
            Neurons.ForEach(q => 
            {
                var weightedInput = q.Weights.Select((w, j) => value[j] * w.Value).Sum();
                q.Output = _activator.CalculateValue(weightedInput + q.Bias.Value);
            });

            return Neurons.Select(q => q.Output).ToArray();
        }

        public override double[] PassBackward(double[] value)
        {
            double[] deltas = new double[NumberOfNeuronsInPreviouseLayer];
            Neurons.ForEach((q, i) => q.Delta = value[i] * _activator.CalculateDeriviative(q.Output));
            Neurons.ForEach(q => q.Weights.ForEach((w, i) => deltas[i] += w.Value * q.Delta));
            return deltas;
        }
    }
}
