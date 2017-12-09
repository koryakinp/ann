namespace Ann.Connections
{
    public class Connection
    {
        public readonly Coordinate Prev;
        public readonly Coordinate Next;

        public Connection(Coordinate left, Coordinate right)
        {
            Prev = left;
            Next = right;
        }

        public double Weight { get; set; }
    }
}
