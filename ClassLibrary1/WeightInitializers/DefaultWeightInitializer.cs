using System;
using System.Collections.Generic;
using System.Text;

namespace Ann.WeightInitializers
{
    public class DefaultWeightInitializer : IWeightInitializer
    {
        private readonly Random _rnd;

        public DefaultWeightInitializer()
        {
            _rnd = new Random();
        }

        public double InitializeWeight(int numberOfInputs, int numberOfOutputs)
        {
            double magnitutde = 1 / Math.Sqrt(numberOfInputs);
            return _rnd.NextDouble() * magnitutde - magnitutde / 2;
        }
    }
}
