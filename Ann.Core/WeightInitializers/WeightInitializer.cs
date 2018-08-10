using System;

namespace Ann.Core.WeightInitializers
{
    public class WeightInitializer : IWeightInitializer
    {
        public double GenerateRandom(double magnitude)
        {
            return new Random(Guid.NewGuid().GetHashCode()).NextDouble() * magnitude - magnitude / 2;
        }
    }
}
