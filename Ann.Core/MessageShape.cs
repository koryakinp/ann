namespace Ann.Core
{
    public class MessageShape
    {
        public readonly int Size;
        public readonly int Depth;

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
