using System.Collections.Generic;

namespace Ann.Model
{
    internal class LayerModel
    {
        public List<NeuronModel> Neurons { get; set; }
        public int LayerIndex { get; set; }
    }
}
