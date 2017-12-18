using System;

namespace Ann.Decayers
{
    public class ExponentialDecayer : ILearningRateDecayer
    {
        private double _initialLearningRate;
        private double _decayRate;

        public ExponentialDecayer(double initialLearningRate, double decayRate)
        {
            _decayRate = decayRate;
            _initialLearningRate = initialLearningRate;
        }

        public double Decay(int epoch)
        {
            return _initialLearningRate * Math.Pow(Math.E, -_decayRate * epoch);
        }
    }
}
