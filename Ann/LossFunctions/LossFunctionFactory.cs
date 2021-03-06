﻿using System;

namespace Ann.LossFunctions
{
    internal static class LossFunctionFactory
    {
        public static LossFunction Produce(LossFunctionType type)
        {
            switch(type)
            {
                case LossFunctionType.CrossEntropy: return new CrossEntropyLoss();
                default: throw new Exception("Cost function is not supported");
            }
        }
    }
}
