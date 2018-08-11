﻿using Ann.Utils;
using System;
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

        public static double[,] Create2DInput(byte[] values)
        {
            int size = (int)Math.Sqrt(values.Length);

            var output = new double[size, size];

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    output[i, j] = (double)values[i * size + j] / 255;
                }
            }

            return output;
        }

        public static double[] Create1DInput(byte[] values)
        {
            return values.Select((q, i) => (double)q / 255).ToArray();
        }

        public static byte IntegerFromOutput(double[] values)
        {
            return (byte)(values.ToList().IndexOf(values.Max()));
        }
    }
}
