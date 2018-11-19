using Newtonsoft.Json;
using System;

namespace Ann
{
    [Serializable]
    internal class MessageShape
    {
        public readonly int Size;
        public readonly int Depth;

        [JsonConstructor]
        public MessageShape(int size, int depth)
        {
            Size = size;
            Depth = depth;
        }

        public MessageShape(int size) : this(size, 1) { }

        public int GetLength()
        {
            return Size * Depth;
        }
    }
}
