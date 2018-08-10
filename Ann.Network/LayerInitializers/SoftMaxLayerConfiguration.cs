using System.Linq;
using Ann.Core;
using Ann.Core.Layers.SoftMax;
using Gdo;

namespace Ann.Network.LayerInitializers
{
    public class SoftMaxLayerConfiguration : LayerConfiguration
    {
        private readonly Optimizer _optimizer;

        public SoftMaxLayerConfiguration(Optimizer optimizer)
        {
            _optimizer = optimizer;
        }

        public override ILayer CreateLayer(Network network)
        {
            return new SoftMaxLayer(
                network._layers.Last().GetNumberOfOutputs(),
                network._numberOfClasses,
                _optimizer);
        }
    }
}
