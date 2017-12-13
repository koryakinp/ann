using Ann.Resources;
using System;

namespace Ann.Configuration
{
    public class NetworkConfiguration
    {
        public readonly LayerConfiguration LayerConfiguration;
        public readonly double LearningRate;
        public readonly double Momentum;

        /// <summary>
        /// Creates a NetworkConfiguration object based in provided LayerConfiguration and optional learning rate and momentum
        /// </summary>
        /// <param name="layerConfiguration">Layer configuration</param>
        /// <param name="learningRate">Learning rate</param>
        /// <param name="momentum">Momentum</param>
        public NetworkConfiguration(LayerConfiguration layerConfiguration, double learningRate = 0.10, double momentum = 0.9)
        {
            if(learningRate <= 0 || learningRate > 1)
            {
                throw new Exception(Messages.InvalidLearningRate);
            }
            else if(momentum < 0 || momentum > 1)
            {
                throw new Exception(Messages.InvalidMomentum);
            }
            LearningRate = learningRate;
            Momentum = momentum;

            LayerConfiguration = layerConfiguration;
        }
    }
}
