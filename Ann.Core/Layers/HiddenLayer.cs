using Gdo;
using System.Linq;
using Activator = Ann.Activators.Activator;
using System;
using MathNet.Numerics.LinearAlgebra.Double;

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

        public override Array PassForward(Array input)
        {
            PrevLayerOutput = input;

            foreach (var neuron in Neurons)
            {
                var vector1 = new DenseVector(input.OfType<double>().ToArray());
                var vector2 = new DenseVector(neuron.Weights.Select(q => q.Value).ToArray());
                var weightedInput = vector1.DotProduct(vector2);
                neuron.Output = _activator.CalculateValue(weightedInput + neuron.Bias.Value);
            }

            return Neurons.Select(q => q.Output).ToArray();
        }

        public override Array PassBackward(Array error)
        {
            double[] deltas = new double[InputMessageShape.Size];
            for (int i = 0; i < Neurons.Count; i++)
            {
                var neuron = Neurons[i];
                neuron.Delta = (double)error.GetValue(i) * _activator.CalculateDeriviative(neuron.Output);

                for (int j = 0; j < neuron.Weights.Length; j++)
                {
                    var weight = neuron.Weights[j];
                    deltas[j] += weight.Value * neuron.Delta;
                }
            }
            return deltas;
        }
    }
}
