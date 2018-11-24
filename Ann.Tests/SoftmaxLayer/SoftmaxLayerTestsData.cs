namespace Ann.Core.Tests.SoftmaxLayer
{
    public static class SoftmaxLayerTestsData
    {
        public static readonly double[][] ForwardPassInput;
        public static readonly double[][] ForwardPassOutput;
        public static readonly double[][] BackwardPassInput;
        public static readonly double[][] BackwardPassOutput;

        static SoftmaxLayerTestsData()
        {
            ForwardPassInput = new double[3][]
            {
                new double[] { -2.2925,-1.1879,-2.3379 },
                new double[] { 1.4611,0.0585,1.4158 },
                new double[] { -2.2365,-1.8739,0.9003 }
            };

            ForwardPassOutput = new double[3][]
            {
                new double[] { 0.201060304,0.606803421,0.192136275 },
                new double[] { 0.454201217,0.111713807,0.434084975 },
                new double[] { 0.039266327,0.056428103,0.90430557 }
            };

            BackwardPassInput = new double[3][]
            {
                new double[] { 0.5167,0.7244,-0.8675 },
                new double[] { -0.9848,0.2259,0.634 },
                new double[] { 0.7061,0.6734,0.6722 }
            };

            BackwardPassOutput = new double[3][]
            {
                new double[] { 0.02813,0.21094,-0.23907 },
                new double[] { -0.38060,0.04164,0.33896 },
                new double[] { 0.00128,-0.00001,-0.00126 }
            };
        }
    }
}
