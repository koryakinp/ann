using Ann.Utils;
using Gdo;
using System.Linq;
using Activator = Ann.Activators.Activator;
using static Ann.Utils.Extensions;

namespace Ann.Core.Layers
{
    public class HiddenLayer : NeuronLayer
    {
        private readonly Activator _activator;

        public HiddenLayer(
            int numberOfNeurons,
            Activator activator,
            Optimizer optimizer,
            MessageShape inputMessageShape) 
            : base(numberOfNeurons, inputMessageShape, optimizer)
        {
            _activator = activator;
        }

        public override Message PassForward(Message input)
        {
            PrevLayerOutput = input.ToSingle();

            Neurons.ForEach(q =>
            {
                var weightedInput = q.Weights.Select((w, j) => PrevLayerOutput[j] * w.Value).Sum();
                q.Output = _activator.CalculateValue(weightedInput + q.Bias.Value);
            });

            return new Message(Neurons.Select(q => q.Output).ToArray());
        }

        public override Message PassBackward(Message error)
        {
            var value = error.ToSingle();

            double[] deltas = new double[InputMessageShape.Height];
            Neurons.ForEach((q, i) => q.Delta = value[i] * _activator.CalculateDeriviative(q.Output));
            Neurons.ForEach(q => q.Weights.ForEach((w, i) => deltas[i] += w.Value * q.Delta));
            return new Message(deltas);
        }
    }
}
