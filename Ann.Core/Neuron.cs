using Ann.Core.WeightInitializers;
using Ann.Utils;
using Gdo;
using System;

namespace Ann.Core
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

        public void RandomizeWeights(IWeightInitializer initializer)
        {
            if(Weights.Length != Weights.Length)
            {
                throw new Exception(Consts.CanNotSetWeights);
            }

            var magnitude = (double)Decimal.Divide(new decimal(1), new decimal(Weights.Length));
            Weights.ForEach((w, i) => Weights[i].Value = initializer.GenerateRandom(magnitude));
        }

        public void UpdateWeights(double[] values)
        {
            if (values.Length != Weights.Length)
            {
                throw new Exception(Consts.CanNotUpdateWeights);
            }

            Weights.ForEach((q, i) => Weights[i].Update(Delta * values[i]));
        }

        public void UpdateBias()
        {
            Bias.Update(Delta);
        }
    }
}
