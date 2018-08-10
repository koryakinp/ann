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

        public override Message PassBackward(Message input)
        {
            var errors = input.ToSingle();
            Neurons.ForEach((q, i) => q.Delta = 0);
            for (int i = 0; i < Neurons.Count; i++)
            {
                for (int j = 0; j < Neurons.Count; j++)
                {
                    var o1 = Neurons[i].Output;
                    var o2 = Neurons[j].Output;
                    var er = errors[j];
                    var ds = i == j ? o1 * (1 - o1) : -o1*o2;
                    Neurons[i].Delta += er * ds;
                }
            }

            double[] output = new double[InputMessageShape.Height];
            Neurons.ForEach(q => q.Weights.ForEach((w, i) => output[i] += w.Value * q.Delta));
            return new Message(output);
        }

        public override Message PassForward(Message input)
        {
            var temp = new double[Neurons.Count];
            PrevLayerOutput = input.ToSingle();

            for (int i = 0; i < Neurons.Count; i++)
            {
                temp[i] = Math.Exp(Neurons[i].Weights.Select((q, j) => q.Value * PrevLayerOutput[j]).Sum());
            }

            Neurons.ForEach((q, i) => q.Output = temp[i] / temp.Sum());
            return new Message(Neurons.Select(q => q.Output).ToArray());
        }
    }
}
