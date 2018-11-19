namespace Ann.Core.Tests.HiddenLayer
{
    public static class HiddenLayerTestsData
    {
        public static readonly double[][] ForwardPassInput;
        public static readonly double[][] ForwardPassOutput;
        public static readonly double[][] BackwardPassInput;
        public static readonly double[][] BackwardPassOutput;
        public static readonly double[][][] Weights;
        public static readonly double[][] WeightsUpdated;
        public static readonly double[][] BiasesUpdated;

        static HiddenLayerTestsData()
        {
            ForwardPassInput = new double[3][]
            {
                new double[] { -0.7363,0.054,-0.0756,-0.0374 },
                new double[] { -0.2378,0.5526,-0.5741,0.467 },
                new double[] { -0.9623,0.2476,-0.8467,-0.4359 },
            };

            ForwardPassOutput = new double[3][]
            {
                new double[] { 0.5363,0.5698,0.6616 },
                new double[] { 0.2704,0.4392,0.3978 },
                new double[] { 0.1663,0.3388,0.6686 },
            };

            BackwardPassInput = new double[3][]
            {
                new double[] { -0.1629,-0.2378,0.0048 },
                new double[] { -0.6614,0.2607,0.5034 },
                new double[] { -0.0988,-0.6806,0.1448 },
            };

            BackwardPassOutput = new double[3][]
            {
                new double[] { 0.0278,-0.0071,-0.0436,-0.0489 },
                new double[] { 0.1346,-0.4434,1.0886,-0.5459 },
                new double[] { 0.458,-0.086,0.4441,0.731 },
            };

            Weights = new double[3][][]
            {
                new double[3][]
                {
                    new double[4] {-0.2609,-0.0114,0.1582,0.9075},
                    new double[4] {-0.489,0.1426,0.8314,0.6331 },
                    new double[4] {-0.9013,-0.4255,-0.3673,-0.0508 }
                },
                new double[3][]
                {
                    new double[4] { 0.2376,-0.5099,0.6568,-0.5939 },
                    new double[4] { 0.0095,0.6412,0.3328,-0.8684 },
                    new double[4] { 0.5943,-0.9299,-0.8659,-0.5494 }
                },
                new double[3][]
                {
                    new double[4] { 0.4408,0.1771,0.9448,0.9901 },
                    new double[4] { 0.3612,-0.2224,0.0836,0.4475 },
                    new double[4] { 0.0141,0.3019,-0.4075,-0.6783 }
                }
            };

            WeightsUpdated = new double[3][]
            {
                new double[] { -0.2624,-0.0113,0.158,0.9074,-0.4925,0.1429,0.831,0.6329,-0.9012,-0.4255,-0.3673,-0.0508 },
                new double[] { 0.2687,-0.5822,0.7319,-0.655,0.0076,0.6456,0.3282,-0.8647,0.5873,-0.9136,-0.8829,-0.5356 },
                new double[] { 0.4808,0.1668,0.98,1.0082,0.4343,-0.2412,0.1479,0.4806,0.017,0.3011,-0.4049,-0.677 }
            };

            BiasesUpdated = new double[3][]
            {
                new double[] { 0.002026309, 0.004806771, -0.000106076 },
                new double[] { -0.130833518, 0.007933955, 0.029528107 },
                new double[] { -0.04159511, -0.075913337, -0.003029841 }
            };
        }
    }
}
