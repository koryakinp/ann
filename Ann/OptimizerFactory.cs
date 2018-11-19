using Gdo;
using Gdo.Optimizers;

namespace Ann
{
    public static class Optimizers
    {
        public static Optimizer Flat(double learningRate)
        {
            return new Flat(learningRate);
        }

        public static Optimizer Adam(double learningRate, int period1, int period2)
        {
            return new Adam(learningRate, period1, period2);
        }

        public static Optimizer AdaDelta(double learningRate, int period)
        {
            return new Adadelta(learningRate, period);
        }

        public static Optimizer Adagrad(double learningRate)
        {
            return new Adagrad(learningRate);
        }

        public static Optimizer RMSProp(double learningRate, int period)
        {
            return new RMSprop(learningRate, period);
        }
    }
}
