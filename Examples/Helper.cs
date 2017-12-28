using System.Collections.Generic;
using System.Linq;

namespace Ann.Mnist
{
    public static class Helper
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

        public static List<double> CreateInput(byte[] values)
        {
            return values.Select(q => (double)q / 255).ToList();
        }

        public static byte IntegerFromOutput(List<double> values)
        {
            return (byte)(values.IndexOf(values.Max()));
        }
    }
}
