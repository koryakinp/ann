using Ann.Core.WeightInitializers;

namespace Ann.Core
{
    public interface ILearnable
    {
        void UpdateWeights();
        void UpdateBiases();
        void RandomizeWeights(IWeightInitializer weightInitializer);
    }
}
