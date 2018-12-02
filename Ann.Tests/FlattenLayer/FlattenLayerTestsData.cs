﻿namespace Ann.Core.Tests.FlattenLayer
{
    public static class FlattenLayerTestsData
    {
        public static readonly double[,,] ForwardPassInput;
        public static readonly double[] BackwardPassInput;

        static FlattenLayerTestsData()
        {
            ForwardPassInput = new double[3, 5, 5]
            {
                {
                    { 111, 112, 113, 114, 115 },
                    { 121, 122, 123, 124, 125 },
                    { 131, 132, 133, 134, 135 },
                    { 141, 142, 143, 144, 145 },
                    { 151, 152, 153, 154, 155 }
                },
                {
                    { 211, 212, 213, 214, 215 },
                    { 221, 222, 223, 224, 225 },
                    { 231, 232, 233, 234, 235 },
                    { 241, 242, 243, 244, 245 },
                    { 251, 252, 253, 254, 255 }
                },
                {
                    { 311, 312, 313, 314, 315 },
                    { 321, 322, 323, 324, 325 },
                    { 331, 332, 333, 334, 335 },
                    { 341, 342, 343, 344, 345 },
                    { 351, 352, 353, 354, 355 }
                },
            };

            BackwardPassInput = new double[]
            {
                 111, 211, 311,
                 112, 212, 312,
                 113, 213, 313,
                 114, 214, 314,
                 115, 215, 315,

                 121, 221, 321,
                 122, 222, 322,
                 123, 223, 323,
                 124, 224, 324,
                 125, 225, 325,

                 131, 231, 331,
                 132, 232, 332,
                 133, 233, 333,
                 134, 234, 334,
                 135, 235, 335,

                 141, 241, 341,
                 142, 242, 342,
                 143, 243, 343,
                 144, 244, 344,
                 145, 245, 345,

                 151, 251, 351,
                 152, 252, 352,
                 153, 253, 353,
                 154, 254, 354,
                 155, 255, 355,
            };
        }

    }
}
