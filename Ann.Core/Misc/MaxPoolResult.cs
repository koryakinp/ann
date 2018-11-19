namespace Ann.Misc
{
    public class MaxPoolResult
    {
        public readonly double[,,] Values;
        public readonly bool[,,] Cache;

        public MaxPoolResult(double[,,] values, bool[,,] cache)
        {
            Values = values;
            Cache = cache;
        }
    }
}
