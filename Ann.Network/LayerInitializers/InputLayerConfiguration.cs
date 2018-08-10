using Ann.Core;
using Ann.Core.Layers.Input;

namespace Ann.Network.LayerInitializers
{
    public class InputLayerConfiguration : LayerConfiguration
    {
        private readonly int _size;
        private readonly int _channels;

        public InputLayerConfiguration(int size, int channels)
        {
            _size = size;
            _channels = channels;
        }

        public override ILayer CreateLayer(Network network)
        {
            return new InputLayer(_size, _channels);
        }
    }
}
