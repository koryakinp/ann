using System;
using System.Collections.Generic;
using System.Text;

namespace Ann.Core.Misc
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
