using Ann.Utils;
using Gdo;

namespace Ann.Misc
{
    internal static class Helper
    {
        public static Optimizer[][,,] InitializeKernelOptimizers(
            int depth,
            int numberOfkernels, 
            int kernelSize, 
            Optimizer optimizer)
        {
            var optimizers = new Optimizer[numberOfkernels][,,];
            optimizers.UpdateForEach<Optimizer[,,]>(q => new Optimizer[depth, kernelSize, kernelSize]);
            optimizers.ForEach(q => q.UpdateForEach<Optimizer>(w => optimizer.Clone() as Optimizer));
            return optimizers;
        }

        public static Optimizer[] InitializeBiasOptimizers(
            int numberOfkernels,
            Optimizer optimizer)
        {
            var optimizers = new Optimizer[numberOfkernels];
            optimizers.UpdateForEach<Optimizer>(q => optimizer.Clone() as Optimizer);
            return optimizers;
        }
    }
}
