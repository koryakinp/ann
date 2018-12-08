using System;

namespace Ann
{
    internal interface ILearnable
    {
        void Update();
        void RandomizeWeights(double stddev);
        void SetWeights(Array array);
        void SetBiases(double[] array);
    }
}
