using System;

namespace Ann
{
    public interface ILearnable
    {
        void UpdateWeights();
        void UpdateBiases();
        void RandomizeWeights(double stddev);
        void SetWeights(Array array);
        void SetBiases(Array array);
    }
}
