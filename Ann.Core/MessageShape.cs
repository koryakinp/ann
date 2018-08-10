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

        public int GetLength()
        {
            return Width * Height * Depth;
        }
    }
}
