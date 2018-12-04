using Gdo;
using System;

namespace Ann.Misc
{
    public class PlaceholderOptimizer : Optimizer
    {
        public PlaceholderOptimizer() { }

        public override void Update(double dx)
        {
            throw new Exception(Consts.CanNotCallPlaceholderOptimizer);
        }
    }
}