using Ann.Core.WeightInitializers;
using Ann.Utils;
using Gdo;

namespace Ann.Core
{
    public class Kernel
    {
        public readonly Optimizer[,,] Weights;
        public readonly double[,,] Gradients;
        public readonly Optimizer Bias;

        public Kernel(int size, int depth, Optimizer optimizer)
        {
            Gradients = new double[size, size, depth];
            Weights = new Optimizer[size, size, depth];
            Bias = optimizer.Clone() as Optimizer;
            Weights.UpdateForEach<Optimizer>(q => optimizer.Clone() as Optimizer);
        }

        public void RandomizeWeights(IWeightInitializer weightInitializer)
        {
            double magnitude = 1 / (Weights.GetLength(0) * Weights.GetLength(1) * Weights.GetLength(2));
            Weights.ForEach((q) => q.SetValue(weightInitializer.GenerateRandom(magnitude)));
        }

        public double[,,] GetValues()
        {
            var output = new double[
                Weights.GetLength(0),
                Weights.GetLength(1),
                Weights.GetLength(2)];

            Weights.ForEach((q, i, j, k) => output[i, j, k] = q.Value);

            return output;
        }
    }
}
