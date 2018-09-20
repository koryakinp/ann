namespace Ann.Core
{
    public class MessageShape
    {
        public readonly int Width;
        public readonly int Height;
        public readonly int Depth;

        public MessageShape(int width, int height, int depth)
        {
            Width = width;
            Height = height;
            Depth = depth;
        }

        public MessageShape(int size) : this(1, size, 1) { }

        public int GetLength()
        {
            return Width * Height * Depth;
        }
    }
}
