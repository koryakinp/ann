namespace Ann.Decayers
{
    public class StepDecayer : ILearningRateDecayer
    {
        private double _initialLearningRate;
        private readonly double _decayRate;
        private readonly int _period;

        public StepDecayer(double initialLearningRate, double decayRate, int period)
        {
            _decayRate = decayRate;
            _period = period;
            _initialLearningRate = initialLearningRate;
        }

        public double Decay(int epoch)
        {
            if(epoch % _period == 0)
            {
                _initialLearningRate = _initialLearningRate * _decayRate;
            }

            return _initialLearningRate;
        }
    }
}
