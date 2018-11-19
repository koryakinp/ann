namespace Ann.Core.Tests.SoftmaxLayer
{
    public static class SoftmaxLayerTestsData
    {
        public static readonly double[][] ForwardPassInput;
        public static readonly double[][] ForwardPassOutput;
        public static readonly double[][] BackwardPassInput;
        public static readonly double[][] BackwardPassOutput;
        public static readonly double[][][] Weights;

        static SoftmaxLayerTestsData()
        {
            Weights = new double[3][][]
            {
                new double[][]
                {
                    new double[] { 0.1582,0.9075,-0.489,0.1426 },
                    new double[] { 0.8314,0.6331,-0.9013,-0.4255 },
                    new double[] { -0.3673,-0.0508,0.3307,0.3973 }
                },
                new double[][]
                {
                    new double[] { 0.6568,-0.5939,0.0095,0.6412 },
                    new double[] { 0.3328,-0.8684,0.5943,-0.9299 },
                    new double[] { -0.8659,-0.5494,-0.1678,0.9017 },
                },
                new double[][]
                {
                    new double[] { 0.9448,0.9901,0.3612,-0.2224 },
                    new double[] { 0.0836,0.4475,0.0141,0.3019 },
                    new double[] { -0.4075,-0.6783,0.6376,0.8129 }
                }
            };

            ForwardPassInput = new double[3][]
            {
                new double[] { -0.0756,-0.0374,-0.4937,0.65 },
                new double[] { -0.5741,0.467,0.0048,-0.8456 },
                new double[] { -0.8467,-0.4359,0.7107,-0.2795 }
            };

            ForwardPassOutput = new double[3][]
            {
                new double[] { 0.3756,0.3055,0.3189 },
                new double[] { 0.1434,0.5753,0.2813 },
                new double[] { 0.115,0.2038,0.6812 }
            };

            BackwardPassInput = new double[3][]
            {
                new double[] { 0.5167,0.7244,-0.8675 },
                new double[] { -0.9848,0.2259,0.634 },
                new double[] { 0.7061,0.6734,0.6722 }

            };

            BackwardPassOutput = new double[3][]
            {
                new double[] { 0.2891,0.2584,-0.3368,-0.1834 },
                new double[] { -0.211,-0.0034,-0.0035,-0.0189 },
                new double[] { 0.0043,0.005,-0.0006,-0.0032 }
            };
        }
    }
}
