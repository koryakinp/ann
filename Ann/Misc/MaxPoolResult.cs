namespace Ann.Misc
{
    internal class MaxPoolResult
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
