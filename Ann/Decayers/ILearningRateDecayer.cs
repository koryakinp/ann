using System;
using System.Collections.Generic;
using System.Text;

namespace Ann.Decayers
{
    public interface ILearningRateDecayer
    {
        /// <summary>
        /// Called after every training epoch.
        /// </summary>
        /// <param name="epoch">Epoch number</param>
        /// <returns>learning rate to be used during the next epoch</returns>
        double Decay(int epoch);
    }
}
