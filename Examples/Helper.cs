using Ann.Utils;
using System.Linq;

namespace Ann.Mnist
{
    public static class Helper
    {
        public static bool[] CreateTarget(byte value)
        {
            var res = new bool[10];
            res.ForEach((q, i) => res[i] = i == value);
            return res;
        }

        public static double[] CreateInput(byte[] values)
        {
            return values.Select(q => (double)q / 255).ToArray();
        }

        public static byte IntegerFromOutput(double[] values)
        {
            return (byte)(values.ToList().IndexOf(values.Max()));
        }
    }
}
