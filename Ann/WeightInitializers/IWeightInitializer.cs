using System;
using System.Collections.Generic;
using System.Text;

namespace Ann
{
    public interface IWeightInitializer
    {
        double InitializeWeight(int numberOfInputs, int numberOfOutputs);
    }
}
