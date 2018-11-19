﻿using Ann.Utils;

namespace Ann.Core.Tests.ActivationLayer
{
    public static class ActivationLayerTestsData
    {
        public static readonly double[][,,] ForwardPassInput;
        public static readonly double[][,,] ForwardPassOutput;

        public static readonly double[][,,] BackwardPassInput;
        public static readonly double[][,,] BackwardPassOutput;

        static ActivationLayerTestsData()
        {
            ForwardPassInput = new double[3][,,]
            {
                ArrayConverter.Convert1Dto3D
                (
                    new double[] { -0.1373,-0.6652,-0.3023,0.3949,-0.7723,0.8314,-0.522,-0.7727,0.6695,-0.1751,0.3415,-0.2609,0.1327,-0.7253,0.5736,-0.4703,-0.0114,0.6249,-0.4426,-0.1919,0.9343,0.1582,0.917,0.8578,0.0139,0.6703,0.9075,0.6387,-0.5064,0.2652,-0.7363,-0.489,0.4913,0.1366,0.542,0.054,0.1426,-0.277,-0.639,-0.9406,-0.0756,0.8314,0.4752,0.6286,-0.4181,-0.0374,0.6331,-0.6089,-0.9931,-0.3484,-0.4937,-0.9013,-0.2657,-0.3771,-0.8656,0.65,-0.4255,0.2283,-0.1915,-0.6498,0.2652,-0.3673,0.9351,0.7472,-0.0991,0.708,-0.0508,-0.6944,0.1182,-0.6649,0.9512,0.3307,-0.263,0.3841,0.8366 },
                    new int[] { 3,5,5 }
                ),
                ArrayConverter.Convert1Dto3D
                (
                    new double[] { -0.6358,-0.1667,-0.8008,-0.1037,-0.2738,0.3328,-0.0234,0.7288,0.171,-0.6736,-0.1629,0.2376,0.6312,-0.2268,0.075,0.0283,-0.5099,-0.8766,0.0559,0.3066,-0.5613,0.6568,-0.5844,0.3593,0.5124,0.1718,-0.5939,0.1401,-0.0078,-0.2333,-0.2378,0.0095,-0.0072,0.6351,0.0435,0.5526,0.6412,-0.7755,0.8625,-0.4421,-0.5741,0.3328,-0.0234,0.1301,0.0804,0.467,-0.8684,0.8925,-0.4946,-0.8469,0.0048,0.5943,-0.7643,-0.8756,0.6359,-0.8456,-0.9299,-0.2702,-0.69,0.8517,-0.2392,-0.8659,-0.5663,0.2486,0.3994,0.2036,-0.5494,-0.1959,-0.3804,0.8366,0.4468,-0.1678,-0.7615,0.8826,0.3381 },
                    new int[] { 3,5,5 }
                ),
                ArrayConverter.Convert1Dto3D
                (
                    new double[] { 0.9273,-0.1657,0.3804,-0.2122,0.0868,0.0836,0.5328,0.5937,-0.3641,0.6455,0.5804,0.4408,-0.2641,0.2629,0.1538,-0.1159,0.1771,0.7666,-0.5344,0.5661,0.7861,0.9448,0.8946,0.7603,0.0371,-0.9267,0.9901,0.7782,0.0512,0.6432,-0.9623,0.3612,-0.2099,-0.1172,-0.7684,0.2476,-0.2224,0.421,-0.6561,0.4326,-0.8467,0.0836,0.7496,0.7794,-0.851,-0.4359,0.4475,0.7956,-0.1442,-0.3573,0.7107,0.0141,-0.5635,-0.8552,-0.2194,-0.2795,0.3019,-0.9931,0.4232,-0.6856,0.1538,-0.4075,-0.3601,-0.295,0.1983,-0.9956,-0.6783,-0.0062,0.2155,0.3526,0.4413,0.6376,-0.0891,0.61,-0.9085 },
                    new int[] { 3,5,5 }
                ),
            };

            ForwardPassOutput = new double[3][,,]
            {
                ArrayConverter.Convert1Dto3D
                (
                    new double[] { 0.4657,0.3396,0.425,0.5975,0.316,0.6967,0.3724,0.3159,0.6614,0.4563,0.5846,0.4351,0.5331,0.3262,0.6396,0.3845,0.4972,0.6513,0.3911,0.4522,0.7179,0.5395,0.7144,0.7022,0.5035,0.6616,0.7125,0.6545,0.376,0.5659,0.3238,0.3801,0.6204,0.5341,0.6323,0.5135,0.5356,0.4312,0.3455,0.2808,0.4811,0.6967,0.6166,0.6522,0.397,0.4907,0.6532,0.3523,0.2703,0.4138,0.379,0.2888,0.434,0.4068,0.2962,0.657,0.3952,0.5568,0.4523,0.343,0.5659,0.4092,0.7181,0.6786,0.4752,0.67,0.4873,0.3331,0.5295,0.3396,0.7214,0.5819,0.4346,0.5949,0.6977 },
                    new int[] { 3,5,5 }
                ),
                ArrayConverter.Convert1Dto3D
                (
                    new double[] { 0.3462,0.4584,0.3099,0.4741,0.432,0.5824,0.4942,0.6745,0.5426,0.3377,0.4594,0.5591,0.6528,0.4435,0.5187,0.5071,0.3752,0.2939,0.514,0.5761,0.3632,0.6585,0.3579,0.5889,0.6254,0.5428,0.3557,0.535,0.4981,0.4419,0.4408,0.5024,0.4982,0.6536,0.5109,0.6347,0.655,0.3153,0.7032,0.3912,0.3603,0.5824,0.4942,0.5325,0.5201,0.6147,0.2956,0.7094,0.3788,0.3001,0.5012,0.6444,0.3177,0.2941,0.6538,0.3004,0.2829,0.4329,0.334,0.7009,0.4405,0.2961,0.3621,0.5618,0.5985,0.5507,0.366,0.4512,0.406,0.6977,0.6099,0.4581,0.3183,0.7074,0.5837 },
                    new int[] { 3,5,5 }
                ),
                ArrayConverter.Convert1Dto3D
                (
                    new double[] { 0.7165,0.4587,0.594,0.4471,0.5217,0.5209,0.6301,0.6442,0.41,0.656,0.6412,0.6084,0.4344,0.5653,0.5384,0.4711,0.5442,0.6828,0.3695,0.6379,0.687,0.7201,0.7098,0.6814,0.5093,0.2836,0.7291,0.6853,0.5128,0.6555,0.2764,0.5893,0.4477,0.4707,0.3168,0.5616,0.4446,0.6037,0.3416,0.6065,0.3001,0.5209,0.6791,0.6856,0.2992,0.3927,0.61,0.689,0.464,0.4116,0.6706,0.5035,0.3627,0.2983,0.4454,0.4306,0.5749,0.2703,0.6042,0.335,0.5384,0.3995,0.4109,0.4268,0.5494,0.2698,0.3366,0.4985,0.5537,0.5872,0.6086,0.6542,0.4777,0.6479,0.2873 },
                    new int[] { 3,5,5 }
                ),
            };

            BackwardPassInput = new double[3][,,]
            {
                ArrayConverter.Convert1Dto3D
                (
                    new double[] { -0.5571,-0.308,-0.2578,-0.1581,0.662,0.8304,-0.8366,-0.9919,-0.2656,-0.4944,0.2664,0.0191,0.961,-0.7195,-0.8218,0.2108,0.4339,-0.2627,-0.395,-0.4333,0.7081,0.8854,-0.4701,-0.3428,-0.3556,0.1381,0.0353,-0.985,0.3124,0.2244,-0.3304,0.2308,0.9195,0.4786,-0.4679,-0.8487,0.4329,0.9202,-0.9925,-0.7057,0.1353,-0.3403,0.6724,-0.8489,-0.5808,-0.821,0.5459,-0.6486,0.4494,-0.3159,-0.0436,-0.0271,0.1558,0.8314,-0.0832,0.1883,-0.6557,-0.1043,0.0382,-0.6987,0.9576,0.8377,0.7134,0.6284,-0.0022,-0.6583,-0.4368,0.3299,0.6825,0.7599,0.8805,0.9658,0.1755,-0.4973,0.8666 },
                    new int[] { 3,5,5 }
                ),
                ArrayConverter.Convert1Dto3D
                (
                    new double[] { -0.0527,-0.8066,-0.7622,0.3404,-0.8395,-0.6652,0.659,-0.4934,-0.7642,-0.9929,0.7649,0.5176,0.4566,0.782,0.6796,-0.2936,0.9324,-0.7671,-0.8935,-0.9319,0.2037,-0.6161,0.0343,0.1557,0.1429,0.6425,0.5339,-0.4806,0.8109,0.7229,0.174,0.7352,-0.5819,0.9772,0.0306,0.6469,-0.0715,0.4217,0.5089,-0.2071,0.6397,-0.8447,-0.829,0.6526,-0.0823,-0.3224,0.0415,-0.1501,0.948,-0.8144,-0.548,0.4715,-0.3486,0.3328,-0.5817,-0.3103,-0.1513,0.4001,0.5426,0.8028,-0.5439,-0.6638,0.2148,-0.873,-0.5008,0.8373,0.0618,-0.1745,0.184,-0.7416,-0.621,0.4615,0.674,-0.9958,-0.6349 },
                    new int[] { 3,5,5 }
                ),
                ArrayConverter.Convert1Dto3D
                (
                    new double[] { -0.3413,-0.8956,0.631,-0.5801,-0.2061,0.3328,-0.0234,0.7288,0.171,-0.6736,0.1812,-0.263,0.3262,-0.7768,-0.1307,0.6573,0.0474,-0.5381,0.7129,0.7162,-0.8416,0.323,-0.9474,0.1146,-0.8749,-0.443,0.3501,0.1664,0.0633,0.1672,-0.9425,0.6331,-0.1581,-0.2155,0.1302,0.1674,0.5845,0.7674,0.8448,0.8033,0.7619,-0.3864,-0.6417,0.0631,-0.2141,0.847,-0.5259,-0.0056,-0.9937,-0.762,-0.9692,0.2017,0.8299,0.0836,0.0708,0.9928,0.7338,-0.1887,0.2387,-0.7427,0.8427,-0.8909,0.004,0.1629,0.7382,-0.9239,-0.4884,0.5324,0.3547,0.2375,0.1648,0.0391,0.8387,-0.1473,-0.8383 },
                    new int[] { 3,5,5 }
                ),
            };

            BackwardPassOutput = new double[3][,,]
            {
                ArrayConverter.Convert1Dto3D
                (
                    new double[] { -0.1386,-0.0691,-0.063,-0.038,0.1431,0.1755,-0.1955,-0.2144,-0.0595,-0.1227,0.0647,0.0047,0.2392,-0.1581,-0.1894,0.0499,0.1085,-0.0597,-0.0941,-0.1073,0.1434,0.22,-0.0959,-0.0717,-0.0889,0.0309,0.0072,-0.2228,0.0733,0.0551,-0.0723,0.0544,0.2165,0.1191,-0.1088,-0.212,0.1077,0.2257,-0.2244,-0.1425,0.0338,-0.0719,0.159,-0.1926,-0.139,-0.2052,0.1237,-0.148,0.0886,-0.0766,-0.0103,-0.0056,0.0383,0.2006,-0.0173,0.0424,-0.1567,-0.0257,0.0095,-0.1575,0.2352,0.2025,0.1444,0.1371,-0.0005,-0.1456,-0.1091,0.0733,0.17,0.1704,0.177,0.235,0.0431,-0.1198,0.1828 },
                    new int[] { 3,5,5 }
                ),
                ArrayConverter.Convert1Dto3D
                (
                    new double[] { -0.0119,-0.2003,-0.163,0.0849,-0.206,-0.1618,0.1647,-0.1083,-0.1897,-0.2221,0.19,0.1276,0.1035,0.193,0.1697,-0.0734,0.2186,-0.1592,-0.2232,-0.2276,0.0471,-0.1385,0.0079,0.0377,0.0335,0.1594,0.1224,-0.1196,0.2027,0.1783,0.0429,0.1838,-0.1455,0.2212,0.0076,0.15,-0.0162,0.091,0.1062,-0.0493,0.1474,-0.2054,-0.2072,0.1625,-0.0205,-0.0764,0.0086,-0.0309,0.2231,-0.1711,-0.137,0.1081,-0.0756,0.0691,-0.1317,-0.0652,-0.0307,0.0982,0.1207,0.1683,-0.134,-0.1384,0.0496,-0.2149,-0.1203,0.2072,0.0143,-0.0432,0.0444,-0.1564,-0.1478,0.1146,0.1463,-0.2061,-0.1543 },
                    new int[] { 3,5,5 }
                ),
                ArrayConverter.Convert1Dto3D
                (
                    new double[] { -0.0693,-0.2224,0.1522,-0.1434,-0.0514,0.0831,-0.0055,0.167,0.0414,-0.152,0.0417,-0.0627,0.0801,-0.1909,-0.0325,0.1638,0.0118,-0.1165,0.1661,0.1654,-0.181,0.0651,-0.1951,0.0249,-0.2186,-0.09,0.0691,0.0359,0.0158,0.0378,-0.1885,0.1532,-0.0391,-0.0537,0.0282,0.0412,0.1443,0.1836,0.19,0.1917,0.16,-0.0964,-0.1398,0.0136,-0.0449,0.202,-0.1251,-0.0012,-0.2471,-0.1845,-0.2141,0.0504,0.1918,0.0175,0.0175,0.2434,0.1793,-0.0372,0.0571,-0.1655,0.2094,-0.2137,0.001,0.0399,0.1827,-0.182,-0.1091,0.1331,0.0877,0.0576,0.0393,0.0088,0.2093,-0.0336,-0.1717 },
                    new int[] { 3,5,5 }
                ),
            };
        }
    }
}