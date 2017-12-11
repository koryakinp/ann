using System.Collections.Generic;

namespace Ann.Model
{
    public class LayerModel
    {
        public List<NeuronModel> Neurons { get; set; }
        public int LayerIndex { get; set; }
    }
}
