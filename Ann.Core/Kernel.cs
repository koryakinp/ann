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

        public int GetNumberOfChannels => Weights.GetLength(0);

        public void SetGradient(double[,,] gradient)
        {
            gradient.ForEach((q, i, j, k) => Gradients[i, j, k] = q);
        }

        public double[,] GetValuesByChannel(int channel)
        {
            var output = new double[Weights.GetLength(1), Weights.GetLength(2)];
            for (int j = 0; j < Weights.GetLength(1); j++)
            {
                for (int k = 0; k < Weights.GetLength(2); k++)
                {
                    output[j, k] = Weights[channel,j,k].Value;
                }
            }

            return output;
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

        public void UpdateWeights()
        {
            Weights.ForEach((q, i, j, k) => q.Update(Gradients[i, j, k]));
        }

        public void UpdateBias()
        {
            double sum = 0;
            Gradients.ForEach(q => sum += q);
            Bias.Update(sum);
        }
    }
}
