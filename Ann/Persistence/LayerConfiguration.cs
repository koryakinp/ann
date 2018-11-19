using Newtonsoft.Json;

namespace Ann.Persistence
{
    internal abstract class LayerConfiguration
    {
        [JsonProperty]
        public readonly MessageShape MessageShape;

        public LayerConfiguration(MessageShape inputMessageShape)
        { 
            MessageShape = inputMessageShape;
        }
    }
}
