namespace Ann.Connections
{
    internal class Coordinate
    {
        public readonly int NeuronIndex;
        public readonly int LayerIndex;

        public Coordinate(int neuronIndex, int layerIndex)
        {
            NeuronIndex = neuronIndex;
            LayerIndex = layerIndex;
        }
    }
}
