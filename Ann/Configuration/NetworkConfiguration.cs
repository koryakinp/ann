namespace Ann.Configuration
{
    public class NetworkConfiguration
    {
        public readonly LayerConfiguration LayerConfiguration;
        public readonly double LearningRate;
        public readonly double Momentum;

        public NetworkConfiguration(LayerConfiguration layerConfiguration, double learningRate = 0.05, double momentum = 0.9)
        {
            if(learningRate <= 0 || learningRate > 1)
            {

            }
            else if(momentum < 0 || momentum > 1)
            {

            }

            LayerConfiguration = layerConfiguration;
        }
    }
}
