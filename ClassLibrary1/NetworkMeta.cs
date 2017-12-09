namespace Ann
{
    public class NetworkMeta
    {
        public readonly double LearningRate;
        public readonly double Momentum;

        public NetworkMeta(double learningRate, double momentum)
        {
            LearningRate = learningRate;
            Momentum = momentum;
        }
    }
}
