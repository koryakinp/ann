namespace Ann.Core.Tests.HiddenLayer
{
    public static class HiddenLayerTestsData
    {
        public static readonly double[][] ForwardPassInput;
        public static readonly double[][] ForwardPassOutput;
        public static readonly double[][] BackwardPassInput;
        public static readonly double[][] BackwardPassOutput;
        public static readonly double[][] Weights;
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
                new double[] { 0.0381,-0.0083,-0.0553,-0.0737 },
                new double[] { 0.0413,-0.0044,-0.1687,-0.0445 },
                new double[] { -0.0607,0.0412,-0.0388,-0.1036 },
            };

            Weights = new double[3][]
            {
                new double[] { -0.2609,-0.0114,0.1582,0.9075,-0.489,0.1426,0.8314,0.6331,-0.9013,-0.4255,-0.3673,-0.0508 },
                new double[] { 0.2376,-0.5099,0.6568,-0.5939,0.0095,0.6412,0.3328,-0.8684,0.5943,-0.9299,-0.8659,-0.5494 },
                new double[] { 0.4408,0.1771,0.9448,0.9901,0.3612,-0.2224,0.0836,0.4475,0.0141,0.3019,-0.4075,-0.6783 }
            };

            WeightsUpdated = new double[3][]
            {
                new double[] { -0.2639,-0.0112,0.1579,0.9073,-0.4933,0.1429,0.831,0.6329,-0.9012,-0.4255,-0.3673,-0.0508 },
                new double[] { 0.2345,-0.5027,0.6493,-0.5878,0.011,0.6377,0.3365,-0.8714,0.5972,-0.9366,-0.859,-0.555 },
                new double[] { 0.4395,0.1774,0.9436,0.9895,0.3465,-0.2186,0.0707,0.4409,0.0172,0.3011,-0.4048,-0.6769 }
            };

            BiasesUpdated = new double[3][]
            {
                new double[] { 0.0041,0.0058,-0.0001 },
                new double[] { 0.013,-0.0064,-0.0121 },
                new double[] { 0.0014,0.0152,-0.0032 }
            };
        }
    }
}
