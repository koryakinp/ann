using System.Collections.Generic;

namespace Ann.Model
{
    public class NeuronModel
    {
        public string Activator { get; set; }
        public int NeuronIndex { get; set; }
        public List<double> Weights { get; set; }
        public double Bias { get; set; }
    }
}
