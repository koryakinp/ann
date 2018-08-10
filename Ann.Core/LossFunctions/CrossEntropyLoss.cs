using System;

namespace Ann.Core.LossFunctions
{
    public class CrossEntropyLoss : LossFunction
    {
        public override double[] ComputeDeriviative(bool[] target, double[] output)
        {
            var res = new double[target.Length];

            int index = Array.IndexOf(target, true);
            res[index] = -1 / output[index];

            return res;
        }

        public override double ComputeLoss(bool[] target, double[] output)
        {
            int index = Array.IndexOf(target, true);
            return -Math.Log(output[index]);
        }
    }
}
