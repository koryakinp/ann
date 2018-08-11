using Ann.Utils;
using Gdo;
using System;
using System.Linq;

namespace Ann.Core.Layers
{
    public class SoftMaxLayer : NeuronLayer
    {
        public SoftMaxLayer(
            int numberOfNeurons,
            MessageShape inputMessageShape,
            Optimizer optimizer) 
            : base(numberOfNeurons, inputMessageShape, optimizer) {}

        public override Array PassBackward(Array errors)
        {
            Neurons.ForEach((q, i) => q.Delta = 0);
            for (int i = 0; i < Neurons.Count; i++)
            {
                for (int j = 0; j < Neurons.Count; j++)
                {
                    var o1 = Neurons[i].Output;
                    var o2 = Neurons[j].Output;
                    var er = (double)errors.GetValue(j);
                    var ds = i == j ? o1 * (1 - o1) : -o1 * o2;
                    Neurons[i].Delta += er * ds;
                }
            }

            double[] output = new double[InputMessageShape.Height];
            for (int i = 0; i < Neurons.Count; i++)
            {
                var neuron = Neurons[i];
                for (int j = 0; j < neuron.Weights.Length; j++)
                {
                    var weight = neuron.Weights[j];
                    output[j] += weight.Value * neuron.Delta;
                }
            }

            return output;
        }

        public override Array PassForward(Array input)
        {
            var temp = new double[Neurons.Count];

            for (int i = 0; i < Neurons.Count; i++)
            {
                var neuron = Neurons[i];
                double weightedSum = 0;

                for (int j = 0; j < neuron.Weights.Length; j++)
                {
                    var weight = neuron.Weights[j];
                    weightedSum += weight.Value * (double)input.GetValue(j);
                }

                temp[i] = Math.Exp(weightedSum);
            }

            var sum = temp.Sum();
            for (int i = 0; i < Neurons.Count; i++)
            {
                Neurons[i].Output = temp[i] / sum;
            }

            return Neurons.Select(q => q.Output).ToArray();
        }
    }
}
