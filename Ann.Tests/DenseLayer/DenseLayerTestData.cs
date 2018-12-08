using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Ann.Tests.DenseLayer
{
    public static class DenseLayerTestData
    {
        public static readonly double[][] DenseLayerForwardPassInput;
        public static readonly double[][] DenseLayerForwardPassOutput;
        public static readonly double[][] DenseLayerBackwardPassInput;
        public static readonly double[][] DenseLayerBackwardPassOutput;
        public static readonly double[][,] Weights;
        public static readonly double[][] Biases;

        public static readonly double[][,] WeightsUpdated;
        public static readonly double[][] BiasesUpdated;

        static DenseLayerTestData()
        {
            DenseLayerForwardPassInput = new double[3][]
            {
                new double[4] { -0.2609,-0.0114,0.1582, 0.9075 },
                new double[4] { 0.2376,-0.5099,0.6568,-0.5939 },
                new double[4] { 0.4408,0.1771,0.9448,0.9901 }
            };

            DenseLayerForwardPassOutput = new double[3][]
            {
                new double[3] { -0.59386,1.17738,0.45561 },
                new double[3] { 2.28346,-0.79901,-0.12118 },
                new double[3] { 0.24376,1.00316,0.77438 }
            };

            DenseLayerBackwardPassInput = new double[3][]
            {
                new double[3] { -0.9503,0.5582,0.5167 },
                new double[3] { 0.5453,0.0596,-0.9848 },
                new double[3] { 0.9498,-0.7954,0.7061 }
            };

            DenseLayerBackwardPassOutput = new double[3][]
            {
                new double[4] { -0.46567485,0.01822197,0.96768266,0.53560672 },
                new double[4] { 0.63844204,0.28019148,0.64719318,-0.02614031 },
                new double[4] { 0.12467809,0.81686074,0.40510192,-0.3767749 }
            };

            Weights = new double[3][,]
            {
                Matrix.Build.DenseOfArray(new double[,]
                {
                    { 0.4715,-0.3486, -0.5033, -0.5817 },
                    {0.5842,-0.281, 0.5112, 0.6841 },
                    {-0.6652,-0.3023, 0.3949, -0.7723 }
                }).Transpose().ToArray(),
                Matrix.Build.DenseOfArray(new double[,]
                {
                    { 0.97,    -0.8472, 0.9982,  -0.5627 },
                    { -0.9172, -0.7795, 0.0126,  0.1856},
                    {-0.1667, -0.8008, -0.1037, -0.2738}
                }).Transpose().ToArray(),
                Matrix.Build.DenseOfArray(new double[,]
                {
                    { 0.5293, 0.0257, 0.9684, -0.6514 },
                    { 0.3282, -0.6586, 0.4587, -0.2271 },
                    { -0.1657, 0.3804, -0.2122, 0.0868 }
                }).Transpose().ToArray()
            };

            Biases = new double[3][]
            {
                new double[] { 0.1327, 0.6249, 0.917 },
                new double[] { 0.6312, -0.8766, -0.5844 },
                new double[] { -0.2641, 0.7666, 0.8946 }
            };

            WeightsUpdated = new double[3][,]
            {
                Matrix.Build.DenseOfArray(new double[,]
                {
                    {0.446706673,-0.349683342,-0.488266254,-0.495460275 },
                    {0.598763438,-0.280363652,0.502369276,0.63344335 },
                    {-0.651719297,-0.301710962,0.386725806,-0.819190525 }
                }).Transpose().ToArray(),
                Matrix.Build.DenseOfArray(new double[,]
                {
                    { 0.957043672,-0.819395153,0.962384696,-0.530314633 },
                    { -0.918616096,-0.776460996,0.008685472,0.189139644 },
                    {-0.143301152,-0.851014952,-0.039018336,-0.332287272 }
                }).Transpose().ToArray(),
                Matrix.Build.DenseOfArray(new double[,]
                {
                    { 0.487432816,0.008879042,0.878662896,-0.745439698 },
                    { 0.363261232,-0.644513466,0.533849392,-0.148347446 },
                    { -0.196824888,0.367894969,-0.278912328,0.016889039 }
                }).Transpose().ToArray()
            };

            BiasesUpdated = new double[3][]
            {
                new double[] { 0.22773,0.56908,0.86533 },
                new double[] { 0.57667,-0.88256,-0.48592 },
                new double[] { -0.35908,0.84614,0.82399 }
            };
        }
    }
}
