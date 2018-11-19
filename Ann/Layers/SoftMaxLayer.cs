using Ann.Persistence;
using Ann.Persistence.LayerConfig;
using Ann.Utils;
using Gdo;
using Gdo.Optimizers;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Linq;

namespace Ann.Layers
{
    public class SoftMaxLayer : NeuronLayer
    {
        public SoftMaxLayer(
            int numberOfNeurons,
            MessageShape inputMessageShape,
            Optimizer optimizer) 
            : base(numberOfNeurons, inputMessageShape, optimizer) {}

        internal SoftMaxLayer(SoftmaxLayerConfiguration config)
            : base(config.NumberOfNeurons, config.MessageShape, new Flat(0.1)) { }

        public override LayerConfiguration GetLayerConfiguration()
        {
            return new SoftmaxLayerConfiguration(Neurons.Count, GetWeights(), InputMessageShape);
        }

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

            var W = GetWeightMatrix();
            var dEdX = Neurons.Select(q => q.Delta).ToArray();
            var dEdO = Matrix.Build.Dense(1, dEdX.Length, dEdX);
            return dEdO.Multiply(W.Transpose()).Row(0).ToArray();
        }

        public override Array PassForward(Array input)
        {
            PrevLayerOutput = input;

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
