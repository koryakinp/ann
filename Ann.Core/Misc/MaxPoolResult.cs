using System;
using System.Collections.Generic;
using System.Text;

namespace Ann.Core.Misc
{
    public class MaxPoolResult
    {
        public readonly double[,,] Values;
        public readonly byte[,,] Cache;

        public MaxPoolResult(double[,,] values, byte[,,] cache)
        {
            Values = values;
            Cache = cache;
        }
    }
}
