using System;
using System.Collections.Generic;
using System.Text;

namespace Ann.Core.Tests.NetworkTests
{
    public static class NetworkTestsData
    {
        public static readonly double[,,] Input;
        public static readonly double[][,,] Conv1Weights;
        public static readonly double[][,,] Conv2Weights;
        public static readonly double[][] Dense1Weights;
        public static readonly double[][] Dense2Weights;
        public static readonly bool[] Labels;

        public static readonly double[][,,] Conv1WeightsUpdated;
        public static readonly double[][,,] Conv2WeightsUpdated;
        public static readonly double[][] Dense1WeightsUpdated;
        public static readonly double[][] Dense2WeightsUpdated;

        public static readonly double[] Conv1BiasesUpdated;
        public static readonly double[] Conv2BiasesUpdated;
        public static readonly double[] Dense1BiasesUpdated;

        static NetworkTestsData()
        {
            Input = new double[1, 7, 7]
            {
                {
                    {   -1.8,   -5.2,   -4.1,   -8.7,   -9.5,   3.4,    2.6     },
                    {   1.6,    -1.0,   -7.9,   1.6,    2.4,    -1.8,   -9.0    },
                    {   -1.1,   4.8,    0.3,    -8.7,   -1.4,   5.0,    -1.0    },
                    {   -1.7,   -.7,    1.2,    1.5,    -1.0,   -1.6,   1.1     },
                    {   1.7,    2.1,    -.5,    -1.0,   -1.9,   -1.1,   -1.1    },
                    {   -2.3,   -8.3,   3.6,    -6.2,   -5.4,   -1.7,   1.7     },
                    {   3.2,    1.5,    -1.3,   -1.9,   -6.8,   6.7,    -1.5    }
                }
            };

            Labels = new bool[] { false, true, false };

            Conv1Weights = new double[][,,]
            {
                new double[,,]
                {
                    {
                        { 0.07, 0.09 },
                        { 0.11, -0.06 }
                    }
                },
                new double[,,]
                {
                    {
                        { -0.13, 0.07 },
                        { -0.15, 0.02 }
                    }
                }
            };

            Conv2Weights = new double[][,,]
            {
                new double[,,]
                {
                    {
                        { 0.07,  0.09 },
                        { 0.11, -0.06 }
                    },
                    {
                        { -0.13, 0.07 },
                        { -0.15, 0.02 }
                    },
                },
                new double[,,]
                {
                    {
                        { 0.13, 0.24 },
                        { -0.19, -0.02 }
                    },
                    {
                        { 0.07, 0.09 },
                        { 0.11, -0.06 }
                    }
                },
                new double[,,]
                {
                    {
                        { -0.13, 0.07 },
                        { -0.15, 0.02 }
                    },
                    {
                        { 0.13, 0.24 },
                        { -0.19, -0.02 }
                    }
                }
            };

            Dense1Weights = new double[][]
            {
                new double[] { 0.13,    0.03,    -0.19 },
                new double[] { 0.14,    -0.35,   -0.02 },
                new double[] { -0.41,   0.23,    0.41 },
                new double[] { -0.11,   0.25,    0.59 },
                new double[] { 0.13,    0.25,    0.35 },
            };

            Dense2Weights = new double[][]
            {
                new double[] { 0.11,    -0.11,   0.13,  0.22,    0.21 },
                new double[] { -0.02,   0.13,    0.33,  -0.45,   -0.12},
                new double[] { 0.82,    0.22,    -0.13, 0.22,    0.41},
            };

            Conv1WeightsUpdated = new double[][,,]
            {
                new double[,,]
                {
                    {
                        { 0.04509,   0.08809 },
                        { 0.11711,   -0.03475 }
                    }
                },
                new double[,,]
                {
                    {
                        { -0.09589,  0.03214 },
                        { -0.12382,  0.03524 }
                    }
                }
            };

            Conv2WeightsUpdated = new double[][,,]
            {
                new double[,,]
                {
                    {
                        { 0.07000,    0.09000 },
                        { 0.11000,    -0.06000 }
                    },
                    {
                        { -0.13000,   0.07000 },
                        { -0.15000,   0.02000 }
                    },
                },
                new double[,,]
                {
                    {
                        { 0.12794,   0.22854 },
                        { -0.19315,  -0.02476 }
                    },
                    {
                        { 0.05258,   0.07483 },
                        { 0.09890,   -0.06915 }
                    }
                },
                new double[,,]
                {
                    {
                        { -0.13424,  0.04636 },
                        { -0.15651,  0.01018 }
                    },
                    {
                        { 0.09408,   0.20872 },
                        { -0.21288,  -0.03888 }
                    }
                }
            };

            Dense1WeightsUpdated = new double[][]
            {
                new double[] { 0.13000, 0.03000, -0.19000 },
                new double[] { 0.14000, -0.35000, -0.02000 },
                new double[] { -0.41000, 0.23922, 0.41623 },
                new double[] { -0.11000, 0.23123, 0.57731 },
                new double[] { 0.13000, 0.23797, 0.34187 },
            };

            Dense2WeightsUpdated = new double[][]
            {
                new double[] { 0.11000,-0.11000,0.12285,0.21085,0.20314 },
                new double[] { -0.02000,0.13000,0.34421,-0.43182,-0.10637 },
                new double[] { 0.82000,0.22000,-0.13706,0.21097,0.40323 },
            };

            Dense1BiasesUpdated = new double[]{ 0,0,0.023275645,-0.047384306,-0.030362443 };

            Conv2BiasesUpdated = new double[] { 0, -0.014083289, -0.029040582 };

            Conv1BiasesUpdated = new double[] {  0.00326, -0.00760 };

        }
    }
}
