using Gdo;
using Gdo.Optimizers;
using System;

namespace Ann
{
    internal static class LearningRateAnnealerFactory
    {
        public static Optimizer Produce(LearningRateAnnealerType type)
        {
            switch(type)
            {
                case LearningRateAnnealerType.Adadelta:
                    return new Adadelta(0.1, 100);
                case LearningRateAnnealerType.Adagrad:
                    return new Adagrad(0.1);
                case LearningRateAnnealerType.Adam:
                    return new Adam(0.1, 20, 100);
                case LearningRateAnnealerType.RMSprop:
                    return new RMSprop(0.1, 20);
                default: throw new Exception("Learning annealer type is not supported");
            }
        }
    }
}
