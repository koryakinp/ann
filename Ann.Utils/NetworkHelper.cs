using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Ann.Utils
{
    public static class NetworkHelper
    {
        public static List<double> CreateTarget(byte value)
        {
            List<double> res = new List<double>();

            for (int i = 0; i <= 9; i++)
            {
                res.Add(i == value ? 1 : 0);
            }

            return res;
        }

        public static byte IntegerFromOutput(List<double> values)
        {
            return (byte)(values.IndexOf(values.Max()));
        }
    }
}
