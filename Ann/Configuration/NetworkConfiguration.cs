using Ann.Decayers;

namespace Ann.Configuration
{
    public class NetworkConfiguration
    {
        /// <summary>
        /// Layer Configuration
        /// </summary>
        public readonly LayerConfiguration LayerConfiguration;

        /// <summary>
        /// Learning Rate. Default value 0.1.
        /// </summary>
        public double LearningRate { get; set; }

        /// <summary>
        /// Momentum. Default value 0.
        /// </summary>
        public double Momentum { get; set; }

        /// <summary>
        /// Learning Rate Decay strategy. If not provided a flat learning rate will be used.
        /// </summary>
        public ILearningRateDecayer LearningRateDecayer { get; set; }

        /// <summary>
        /// Creates a NetworkConfiguration object based in provided LayerConfiguration
        /// </summary>
        /// <param name="layerConfiguration">Layer configuration</param>
        public NetworkConfiguration(LayerConfiguration layerConfiguration)
        {
            LearningRate = 0.1;
            Momentum = 0;
            LayerConfiguration = layerConfiguration;
        }
    }
}
