using Newtonsoft.Json;
using System;

namespace Ann.Core.Persistence
{
    //[Serializable]
    public abstract class LayerConfiguration
    {
        [JsonProperty]
        public readonly MessageShape MessageShape;

        public LayerConfiguration(MessageShape inputMessageShape)
        { 
            MessageShape = inputMessageShape;
        }
    }
}
