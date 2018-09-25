using Ann.Core.WeightInitializers;
using Ann.Utils;
using Gdo;

namespace Ann.Core
{
    public class Kernel
    {
        public readonly Optimizer[,,] Weights;
        public readonly Optimizer Bias;

        public Kernel(int size, int depth, Optimizer optimizer)
        {
            Weights = new Optimizer[size, size, depth];
            Bias = optimizer.Clone() as Optimizer;
            Weights.UpdateForEach<Optimizer>(q => optimizer.Clone() as Optimizer);
        }

        public void RandomizeWeights(IWeightInitializer weightInitializer)
        {
            double magnitude = 1 / (Weights.GetLength(0) * Weights.GetLength(1) * Weights.GetLength(2));
            Weights.ForEach((q) => q.SetValue(weightInitializer.GenerateRandom(magnitude)));
        }
    }
}
