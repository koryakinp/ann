using Ann.Core.WeightInitializers;
using System;

namespace Ann.Core
{
    public interface ILearnable
    {
        void UpdateWeights();
        void UpdateBiases();
        void RandomizeWeights(IWeightInitializer weightInitializer);
        void SetWeights(Array array);
    }
}
