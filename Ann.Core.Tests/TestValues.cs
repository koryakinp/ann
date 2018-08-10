using System;
using System.Collections.Generic;
using System.Text;

namespace Ann.Core.Tests
{
    public static class TestValues
    {
        public static readonly List<double[]> Weights;
        public static readonly double[] Input;
        public static readonly double[] Error;
        public static readonly double LR;
        public static readonly double[,,] MultiDimensionalInput;
        public static readonly double[,,] Kernels;

        static TestValues()
        {
            Weights = new List<double[]>()
            {
                new double[] { 0.3, -0.6, 0.4, 0.6 },
                new double[] { 0.2, 0.1, -0.0, 0.2 },
                new double[] { -0.2, 0.8, 0.4, 0.5 }
            };

            Input = new double[] { 0.2, -0.5, -0.3, 0.1 };
            Error = new double[] { 0.1, -0.3, 0.8 };
            LR = 0.1;
            MultiDimensionalInput = new double[2, 5, 5]
            {
                {
                    { 54,  9,   61,  31,  29 },
                    { 91,  81,  80,  59,  18 },
                    { 3,   52,  27,  19,  71 },
                    { 36,  3,   86,  30,  32 },
                    { 2,   56,  53,  59,  91 }
                },
                {
                    { 21,  19,  44,  22,  99 },
                    { 65,  18,  52,  1,   49 },
                    { 90,  27,  5,   12,  41 },
                    { 75,  16,  97,  56,  95 },
                    { 82,  89,  87,  15,  83 }
                }
            };
            Kernels = new double[2, 3, 3]
            {
                {
                    { 2,2,2 },
                    { 2,2,2 },
                    { 2,2,2 }
                },
                {
                    { 3,3,3 },
                    { 3,3,3 },
                    { 3,3,3 }
                }
            };
        }
    }
}
