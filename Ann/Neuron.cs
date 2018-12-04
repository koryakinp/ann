using Ann.Utils;
using Gdo;
using MathNet.Numerics.Distributions;
using System;

namespace Ann
{
    public class Neuron
    {
        public double Output { get; set; }
        public double Delta { get; set; }

        public readonly Optimizer[] Weights;
        public readonly Optimizer Bias;

        public Neuron(Optimizer[] weights, Optimizer bias)
        {
            Weights = weights;
            Bias = bias;
        }

        public void RandomizeWeights(double stddev)
        {
            var dist = new Normal(0, stddev);
            Weights.ForEach((w, i) => Weights[i].Value = dist.Sample());
        }

        public void UpdateWeights(double[] values)
        {
            if (values.Length != Weights.Length)
            {
                throw new Exception(Consts.CanNotUpdateWeights);
            }

            Weights.ForEach((q, i) => 
            {
                var val = Delta * values[i];
                Weights[i].Update(val);
            });
        }

        public void UpdateBias()
        {
            Bias.Update(Delta);
        }
    }
}
