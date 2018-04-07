using Gdo;
using System;

namespace Ann
{
    internal class Neuron
    {
        public double Output { get; set; }
        public double Delta { get; set; }

        public readonly Optimizer[] Weights;
        public readonly Optimizer Bias;

        public Neuron(
            int numberOfConnections,
            LearningRateAnnealerType lrat)
        {
            Weights = new Optimizer[numberOfConnections];
            Weights.ForEach((q, i) => Weights[i] = LearningRateAnnealerFactory.Produce(lrat));
            Bias = LearningRateAnnealerFactory.Produce(lrat);
        }

        public void RandomizeWeights()
        {
            double magnitude = 1 / Math.Sqrt(Weights.Length);
            Weights.ForEach((q, i) => Weights[i].SetValue(RandomGenerator.Generate(magnitude)));
        }

        public void SetWeights(double[] weights)
        {
            Weights.ForEach((q, i) => Weights[i].SetValue(weights[i]));
        }

        public void UpdateWeights(double[] values)
        {
            Weights.ForEach((q, i) => Weights[i].Update(Delta * values[i]));
        }

        public void UpdateBias()
        {
            Bias.Update(Delta);
        }
    }
}
