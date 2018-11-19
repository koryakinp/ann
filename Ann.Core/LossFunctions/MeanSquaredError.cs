using System;
using System.Linq;

namespace Ann.LossFunctions
{
    public class MeanSquaredError : LossFunction
    {
        public override double[] ComputeDeriviative(bool[] target, double[] output)
        {
            return target.Select((q, i) => output[i] - (q ? 1 : 0)).ToArray();
        }

        public override double ComputeLoss(bool[] target, double[] output)
        {
            return output.Select((q, i) => Math.Pow(q - (target[i] ? 1 : 0), 2)).Sum();
        }
    }
}
