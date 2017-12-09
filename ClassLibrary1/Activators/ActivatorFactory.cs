using Ann.Resources;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ann.Activators
{
    public class ActivatorFactory
    {
        public static IActivator Produce(ActivatorType type)
        {
            switch(type)
            {
                case ActivatorType.LogisticActivator: return new LogisticActivator();
                case ActivatorType.TanhActivator: return new TanhActivator();
                case ActivatorType.ReluActivator: return new ReluActivator();
                default: throw new Exception(Messages.ActivatorNotSupported);
            }
        }
    }
}
