namespace Ann.LossFunctions
{
    internal abstract class LossFunction
    {
        public abstract double[] ComputeDeriviative(bool[] target, double[] output);
        public abstract double ComputeLoss(bool[] target, double[] output);
    }
}
