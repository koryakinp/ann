using System;
using System.Collections.Generic;

namespace Ann.Core.Tests.Utils
{
    public class DoubleComparer : Comparer<double>
    {
        private readonly int _precission;

        public DoubleComparer(int precission)
        {
            _precission = precission;
        }

        public override int Compare(double x, double y)
        {
            var res = Math.Abs(x - y);
            if (res < Math.Pow(0.1, _precission))
            {
                return 0;
            }
            else
            {
                return res > 0 ? 1 : -1;
            }
        }
    }
}
