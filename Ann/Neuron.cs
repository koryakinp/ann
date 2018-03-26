using Gdo;
using System;
using Activator = Ann.Activators.Activator;

namespace Ann
{
    internal class Neuron
    {
        public double Output { get; private set; }
        public double Delta { get; private set; }
        private readonly Activator _activator;
        public readonly Optimizer[] Weights;
        public readonly Optimizer Bias;

        public Neuron(
            Activator activator,
            int numberOfConnections,
            LearningRateAnnealerType lrat)
        {
            Weights = new Optimizer[numberOfConnections];
            Weights.ForEach((q, i) => Weights[i] = LearningRateAnnealerFactory.Produce(lrat));
            Bias = LearningRateAnnealerFactory.Produce(lrat);
            _activator = activator;
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

        public void ComputeOutput(double weightedSum)
        {
            Output = _activator.CalculateValue(weightedSum + Bias.Value);
        }

        public void ComputeDelta(double weightedDelta)
        {
            Delta = weightedDelta * _activator.CalculateDeriviative(Output);
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
