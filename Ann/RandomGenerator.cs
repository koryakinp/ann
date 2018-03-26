using System;

namespace Ann
{
    public static class RandomGenerator
    {
        public static double Generate(double magnitude)
        {
            return new Random(Guid.NewGuid().GetHashCode()).NextDouble() * magnitude - magnitude / 2;
        }
    }
}
