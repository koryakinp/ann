using Ann.Core.Tests.Utils;
using Ann.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ann.Core.Tests.Utils
{
    public static class TestValues
    {
        public static readonly List<double[]> Weights;
        public static readonly double[] Input;
        public static readonly double[] Error;
        public static readonly double LR;
        public static readonly double[,,] MultiDimensionalInput;
        public static readonly double[,,] Kernels;
        public static readonly double[,,,] ConvolutionBackwardPassKernels;
        public static readonly double[,,] ConvolutionBackwardPassDeltas;
        public static readonly double[,,] ConvolutionBackwardPassResult;

        public static readonly double[,,] ConvolutionLayerForwardPassInput;
        public static readonly double[,,] ConvolutionLayerForwardPassOutput;
        public static readonly double[,,] ConvolutionLayerBackwardPassInput;
        public static readonly double[,,] ConvolutionLayerBackwardPassOutput;
        public static readonly double[,,,] ConvolutionalLayerKernels;


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
            ConvolutionBackwardPassKernels = ArrayConverter.Convert1Dto4D
            (
                new double[] { -1,1,-1,1,0,0,-1,1,0,1,-1,1,-1,1,0,0,-1,-1,0,1,0,0,0,-1,1,0,0,-1,-1,0,-1,-1,1,0,0,0,-1,-1,0,-1,1,0,0,1,-1,1,-1,-1,1,1,-1,-1,-1,0,-1,0,0,1,-1,-1,1,-1,-1,0,0,-1,1,0,-1,-1,-1,-1,0,0,-1,1,1,-1,-1,-1,-1,0,1,1,1,0,-1,0,1,-1,1,1,1,0,-1,1,-1,0,1,1,0,-1,0,1,0,1,-1,1 },
                new int[] { 4, 3, 3, 3 }
            );
            ConvolutionBackwardPassDeltas = ArrayConverter.Convert1Dto3D
            (
                new double[] { -0.74,0.55,0.65,0.54,0.2,-0.97,-0.51,0.68,0.01,-0.27,-0.91,0.91,0.41,-0.06,0.02,0.54,0.72,0.18,0.61,0.78,0.02,0.13,-0.84,0.66,0.35,0.13,0.59,-0.69,-0.53,0.73,0.97,-0.26,-0.24,0.13,0.81,0.59 },
                new int[] { 4,3,3 }
            );
            ConvolutionBackwardPassResult = ArrayConverter.Convert1Dto3D
            (
                new double[] { 0.4,-1.58,-0.6,-0.61,0.08,-1.63,3.17,1.15,-3.78,0.89,2.2,-5.64,-1.04,2.02,-0.57,-0.98,0.35,1.53,-1.76,-1.42,0.86,-1.28,1.46,-0.93,-1.18,-1.16,1.25,-1.74,-1.59,1.36,2.72,0.84,-2.92,1.01,-1.16,-1.66,-0.52,-1.33,0.92,-1.74,-0.78,-0.53,0.82,1.05,0.05,-0.48,-0.24,-1.52,-1.14,-0.19,-0.96,-1.91,3.45,0.4,-1.66,1.72,-0.67,-0.8,1.41,-2.02,-0.56,1.59,-2.04,-3.66,-0.36,1.86,1.2,2.16,-0.03,-1.68,-1.27,-0.38,-2.05,-0.68,0 },
                new int[] { 3,5,5 }
            );
        }
    }
}
